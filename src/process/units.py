"""Unit conversions. Tracking is in yards and yd/s; SI only where a spec requires it."""
YD_TO_M = 0.9144
LB_TO_KG = 0.45359237


def yd_to_m(x):
    return x * YD_TO_M


def lb_to_kg(x):
    return x * LB_TO_KG
