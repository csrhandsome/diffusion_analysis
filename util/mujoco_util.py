def tune_contact_params(model):
    """调整接触参数以匹配实际力反馈"""
    # 调整接触刚度
    model.opt.elasticity = 1.0
    # 调整阻尼
    model.opt.damping = 0.1
    # 调整摩擦系数
    model.opt.friction = [1.0, 0.005, 0.0001]