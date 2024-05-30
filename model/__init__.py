from models.baseline.hoi import build
from models.dn.hoi_doq import build as build_dn
from models.doq.hoi_doq import build as build_doq
from models.dn_m.hoi_doq import build as build_dn_m
from models.gen.hoi import build as build_gen
from models.gen_dn.hoi import build as build_gen_dn
from models.gen_dn_m.hoi import build as build_gen_dn_m


def build_model(args):
    return build(args)


def build_dn_model(args):
    return build_dn(args)


def build_doq_model(args):
    return build_doq(args)


def build_dn_m_model(args):
    return build_dn_m(args)


def build_gen_model(args):
    return build_gen(args)


def build_gen_dn_model(args):
    return build_gen_dn(args)


def build_gen_dn_m_model(args):
    return build_gen_dn_m(args)
