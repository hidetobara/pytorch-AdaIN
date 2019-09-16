import torch


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_correct(content_feat, style_feat):
    size = content_feat.size()
    N, C, H, W = size

    content_table = content_feat.transpose(1,3).transpose(1,2).view(-1, C)
    style_table = style_feat.transpose(1,3).transpose(1,2).view(-1, C)
    distance = torch.nn.PairwiseDistance()
    table = distance(content_table, style_table).sqrt()
    table_std = table.var(dim=0).sqrt()
    table_mean = table.mean(dim=0)
    table = torch.clamp(1 - (table - table_mean + table_std * 2) / (table_std * 4), min=0.1, max=1.1).view(N, 1, H, W)
    print("table=", table, table_mean, table_std)

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def single_adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    N, C, H, W = size
    style_mean, style_std = calc_mean_std(style_feat)
    normalized_style = (style_feat - style_mean.expand(size)) / style_std.expand(size)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    #return style_feat
    #return normalized_content * normalized_style * style_std.expand(size) + style_mean.expand(size)
    #return normalized_content * style_std.expand(size) + style_mean.expand(size)
    return normalized_content * style_std.expand(size) + style_feat

def correct_adaptive_instance_normalization(content_feat, style_feat, correct_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    N, C, H, W = size
    style_mean, style_std = calc_mean_std(style_feat)
    normalized_style = (style_feat - style_mean.expand(size)) / style_std.expand(size)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    print("cor=", correct_feat.view(H, W))
    correct = correct_feat.expand(N, C, H, W)
    #return style_feat
    #return normalized_content * normalized_style * style_std.expand(size) + style_mean.expand(size)
    #return normalized_content * style_std.expand(size) + style_mean.expand(size)
    return normalized_content * (correct + 0.25) * style_std.expand(size) + style_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
