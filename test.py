from reinforcelearning import Policy, Quantizer


policy = Policy()
policy.load_state_dict(torch.load('/home/liang/PycharmProjects/SourceCoding/model/policy_state_dict'))
policy.eval()


quantizer = Quantizer()
quantizer.load_state_dict(torch.load('/home/liang/PycharmProjects/SourceCoding/model/quantizer_state_dict'))
quantizer.eval()
