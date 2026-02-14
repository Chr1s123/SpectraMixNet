import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------ 融合后的高低频分离处理模块 ------------------
class HighLowFrequencyProcessingBlock(nn.Module):
    def __init__(self, dim, dilation=0, kernel_size=3):
        super(HighLowFrequencyProcessingBlock, self).__init__()
        # 动态卷积参数
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.unfoldSize = kernel_size + dilation * (kernel_size - 1)
        self.conv_filter = nn.Conv2d(dim, dim * kernel_size ** 2, kernel_size=1, bias=False)
        self.norm_filter = nn.GroupNorm(num_groups=4, num_channels=dim * kernel_size ** 2)
        self.act_filter = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv_filter.weight, mode='fan_out', nonlinearity='relu')
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.pad = nn.ReflectionPad2d(self.unfoldSize // 2)

        # 构建unfoldMask（用于低通模式）
        # self.unfoldMask = []
        # for i in range(self.unfoldSize):
        #     for j in range(self.unfoldSize):
        #         if (i % (dilation + 1) == 0) and (j % (dilation + 1) == 0):
        #             self.unfoldMask.append(i * self.unfoldSize + j)
        # 注册为 buffer：unfoldMask
        unfold_mask = []
        for i in range(self.unfoldSize):
            for j in range(self.unfoldSize):
                if (i % (dilation + 1) == 0) and (j % (dilation + 1) == 0):
                    unfold_mask.append(i * self.unfoldSize + j)
        self.register_buffer("unfoldMask", torch.tensor(unfold_mask, dtype=torch.long))

        # 低频处理分支（内联TriScaleConv逻辑）
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.PReLU(),
            # nn.Conv2d(dim // 2, dim // 2, kernel_size=3, dilation=5, padding=5),
            # nn.PReLU(),
            # nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1),
            # nn.PReLU(),
            # nn.Conv2d(dim // 2, dim // 2, kernel_size=3, dilation=3, padding=3),
            # nn.PReLU(),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=3, dilation=5, padding=5, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=1),
            nn.PReLU(),

            # 深度可分离卷积模块2 (替换原3x3卷积)
            nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=1),
            nn.PReLU(),

            # 深度可分离卷积模块3 (替换原3x3膨胀卷积)
            nn.Conv2d(dim // 4, dim // 4, kernel_size=3, dilation=3, padding=3, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=1),
            nn.PReLU(),

            nn.Conv2d(dim // 4, dim, kernel_size=1)
        )

        # 高频处理分支
        self.high_freq_branch = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=dim),
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            # nn.Conv2d(dim, dim, kernel_size=1)
        )

        # 频域融合模块
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
            nn.GroupNorm(num_groups=4, num_channels=dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )

    def forward(self, x):
        # 1. 动态卷积分离高低频
        copy = x
        filter_input = self.ap(x)
        filter_weights = self.conv_filter(filter_input)
        filter_weights = self.norm_filter(filter_weights)

        n, c, h, w = x.shape
        x_unfolded = F.unfold(self.pad(x), kernel_size=self.unfoldSize).reshape(
            n, self.dim, c // self.dim, self.unfoldSize ** 2, h * w
        )
        x_masked = x_unfolded[:, :, :, self.unfoldMask, :]

        n, c1, p, q = filter_weights.shape
        filter_reshaped = filter_weights.reshape(
            n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q
        ).unsqueeze(2)
        filter_softmax = self.act_filter(filter_reshaped)

        low_freq = torch.sum(x_masked * filter_softmax, dim=3).reshape(n, c, h, w)
        high_freq = copy - low_freq

        # 2. 低频分支处理
        enhanced_low = self.low_freq_branch(low_freq)
        # enhanced_low = self.low_freq_branch(low_freq) + low_freq

        # 3. 高频分支处理
        enhanced_high = self.high_freq_branch(high_freq) + high_freq

        # 4. 融合高低频
        fused = torch.cat([enhanced_low, enhanced_high], dim=1)
        output = self.freq_fusion(fused)

        return output


# ------------------ 其他模块保持不变 ------------------
class ECALayer(nn.Module):
    def __init__(self, channels, k_size=None, gamma=2, b=1):
        super(ECALayer, self).__init__()
        if k_size is None:
            t = int(abs((math.log2(channels) + b) / gamma))
            k_size = t if t % 2 else t + 1
            if k_size < 3:
                k_size = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y


class CGAFusion(nn.Module):
    def __init__(self, dim, eca_k_size=None):
        super(CGAFusion, self).__init__()
        self.channel_attn = ECALayer(channels=dim * 2, k_size=eca_k_size)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim * 2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feats):
        x_concat = torch.cat(feats, dim=1)
        attn_c = self.channel_attn(x_concat)
        attn_s = self.spatial_attn(x_concat)
        x_attended = x_concat * attn_c * attn_s
        C_half = x_attended.shape[1] // 2
        f1, f2 = x_attended[:, :C_half], x_attended[:, C_half:]
        return f1 + f2


class MultiScaleSpectralBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        def get_group_norm_groups(channels):
            if channels >= 32:
                candidates = [32, 16, 8, 4, 2, 1]
            elif channels >= 16:
                candidates = [16, 8, 4, 2, 1]
            elif channels >= 8:
                candidates = [8, 4, 2, 1]
            elif channels >= 4:
                candidates = [4, 2, 1]
            else:
                candidates = [1]
            for num_groups in candidates:
                if channels % num_groups == 0:
                    return num_groups
            return 1

        dim_groups = get_group_norm_groups(dim)
        dim2_groups = get_group_norm_groups(dim * 2)

        self.norm1 = nn.GroupNorm(num_groups=dim_groups, num_channels=dim)
        self.norm2 = nn.GroupNorm(num_groups=dim_groups, num_channels=dim)
        self.norm3 = nn.GroupNorm(num_groups=dim_groups, num_channels=dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')

        # 使用融合后的高低频模块
        self.frequency_processor = HighLowFrequencyProcessingBlock(dim)

        self.enhance_conv_s = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3,
                                        padding_mode='reflect')
        self.enhance_conv_m = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3,
                                        padding_mode='reflect')
        self.enhance_conv_l = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3,
                                        padding_mode='reflect')

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

        dim5_groups = get_group_norm_groups(dim * 5)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 5, dim * 2, 1),
            nn.GroupNorm(dim2_groups, dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

        dim3_groups = get_group_norm_groups(dim * 3)
        dim4_groups = get_group_norm_groups(dim * 4)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GroupNorm(dim4_groups, dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

        self.domain_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
            nn.GroupNorm(dim2_groups, dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )

    def forward(self, x):
        identity1 = x
        x_processed = self.norm1(x)
        x_processed = self.conv1(x_processed)
        x_base_feat = self.conv2(x_processed)

        # 使用融合后的高低频模块
        freq_enhanced = self.frequency_processor(x_processed)

        # x_enhanced_s = self.enhance_conv_s(x_base_feat) + x_base_feat
        # x_enhanced_m = self.enhance_conv_m(x_enhanced_s) + x_enhanced_s
        # x_enhanced_l = self.enhance_conv_l(x_enhanced_m) + x_enhanced_m
        x_enhanced_s = self.enhance_conv_s(x_base_feat) + x_base_feat
        x_enhanced_m = self.enhance_conv_m(x_base_feat) + x_base_feat
        x_enhanced_l = self.enhance_conv_l(x_base_feat) + x_base_feat

        x_multi_scale_fused = torch.cat([
            x_base_feat,  # 原始特征
            x_enhanced_s,  # 小尺度增强
            x_enhanced_m,  # 中尺度增强
            x_enhanced_l,  # 大尺度增强
            freq_enhanced  # 高低频模块输出
        ], dim=1)
        x_mlp1_out = self.mlp(x_multi_scale_fused)
        x_after_res1 = identity1 + x_mlp1_out

        identity2 = x_after_res1
        x_norm2 = self.norm2(x_after_res1)

        att_wv_wg = self.Wv(x_norm2) * self.Wg(x_norm2)
        att_ca_weights = self.ca(x_norm2)
        att_ca_out = att_ca_weights * x_norm2
        att_pa_weights = self.pa(x_norm2)
        att_pa_out = att_pa_weights * x_norm2

        x_att_cat = torch.cat([att_wv_wg, att_ca_out, att_pa_out], dim=1)
        x_mlp2_out = self.mlp2(x_att_cat)
        x_after_res2 = identity2 + x_mlp2_out

        identity3 = x_after_res2
        x_norm3 = self.norm3(x_after_res2)
        final_freq_feat = self.frequency_processor(x_norm3)
        domain_fused = torch.cat([x_norm3, final_freq_feat], dim=1)
        final_enhancement = self.domain_fusion(domain_fused)
        x_final = identity3 + final_enhancement

        return x_final


# ------------------ 其他模块保持原样 ------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([MultiScaleSpectralBlock(dim=dim) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.proj(x)


# ------------------ 最终网络结构 ------------------
class SpectraMixNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(SpectraMixNet, self).__init__()
        self.patch_size = 4
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])
        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])
        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])
        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        self.fusion1 = CGAFusion(embed_dims[3])
        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])
        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        self.fusion2 = CGAFusion(embed_dims[4])
        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_embed = self.patch_embed(x)
        l1_out = self.layer1(x_embed)
        skip1_feat_processed = self.skip1(l1_out)
        l2_in = self.patch_merge1(l1_out)
        l2_out = self.layer2(l2_in)
        skip2_feat_processed = self.skip2(l2_out)
        l3_in = self.patch_merge2(l2_out)
        l3_out = self.layer3(l3_in)
        dec1_in = self.patch_split1(l3_out)
        fused1 = self.fusion1([dec1_in, skip2_feat_processed]) + dec1_in
        l4_out = self.layer4(fused1)
        dec2_in = self.patch_split2(l4_out)
        fused2 = self.fusion2([dec2_in, skip1_feat_processed]) + dec2_in
        l5_out = self.layer5(fused2)
        out = self.patch_unembed(l5_out)
        return out

    def forward(self, x):
        H_orig, W_orig = x.shape[2:]
        x_padded = self.check_image_size(x)
        feat = self.forward_features(x_padded)
        K, B = torch.split(feat, (1, 3), dim=1)
        dehazed_img = K * x_padded - B + x_padded
        return dehazed_img[:, :, :H_orig, :W_orig]


def SpectraMixNet_t():
    return SpectraMixNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])