"""Teacher-to-student diagonal Gaussian KL; no simulator labels required."""
import torch


def gaussian_kl(teacher_loc,teacher_scale,student_loc,student_scale):
    assert teacher_loc.shape==teacher_scale.shape==student_loc.shape==student_scale.shape
    assert bool((teacher_scale>0).all() and (student_scale>0).all())
    return (torch.log(student_scale/teacher_scale)+(teacher_scale.square()+(teacher_loc-student_loc).square())/(2*student_scale.square())-.5).sum(-1)


def balanced_branch_loss(kl,recovery):
    kl=kl.reshape(-1);recovery=recovery.reshape(-1).bool()
    assert kl.numel()==recovery.numel() and kl.numel()>0
    terms=[kl[m].mean() for m in [recovery,~recovery] if m.any()]
    return torch.stack(terms).mean()
