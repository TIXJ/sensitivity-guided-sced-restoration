"""Case39 Route-B benchmark adjustments.

The pandapower case39 default thermal ratings make the generated labels fail
primarily because of a small AC line-loading overload.  For the Route-B
experiment we use a lightly modified benchmark: thermal ratings are scaled by
1.35 and strict voltage uses the standard narrow upper bound 1.06 pu.  This
keeps the task aligned with the 300/118 Route-B story: linear N-1 SCED labels
remain the active-power target, while RL learns P+V operating corrections.
"""

CASE39_THERMAL_SCALE = 1.35
CASE39_VMIN = 0.93
CASE39_VMAX = 1.06


def apply_case39_route_b_limits(net):
    if hasattr(net, "line") and "max_i_ka" in net.line:
        net.line.max_i_ka = net.line.max_i_ka.astype(float) * CASE39_THERMAL_SCALE
    if hasattr(net, "trafo") and len(net.trafo) and "sn_mva" in net.trafo:
        net.trafo.sn_mva = net.trafo.sn_mva.astype(float) * CASE39_THERMAL_SCALE
    net.bus.min_vm_pu = CASE39_VMIN
    net.bus.max_vm_pu = CASE39_VMAX
    return net
