
def channelAttention(x,attention_op,):
    """
    to learn the attention of one channel and the rest channels
    :param x:input tensor , shape as B,C,H,W
    :return: output tensor after self-attention op. shape as B,C,H,W
    """
    b,c,h,w = x.shape
    pre_x = x.contiguous().view(b,c,h*w)

    att_x,relations = attention_op(pre_x,pre_x,pre_x)
    att_x = att_x.contiguous().view(b,c,h,w)
    return att_x

def pixelAttention(x,attention_op):
    """
    to learn the attention of one position and the other position
    in spicial scale.
    :param x:
    :param attention_op:
    :return:
    """
    b,c,w,h = x.shape
    pre_x = x.permute(0,2,3,1)
    pre_x = pre_x.contiguous().view(b,h*w,c)
    att_x,relations = attention_op(pre_x,pre_x,pre_x)
    att_x = att_x.contiguous().view(b,h,w,c)
    att_x = att_x.permute(0,3,1,2)
    return att_x

def blockAttention(x,attention_op):
    """
    to learn the relation of n objects
    :param x:
    :param attention_op:
    :return:
    """
    b,n,c,w,h = x.shape
    pre_x = x.contiguous().view(b,n,-1)
    att_x,relations = attention_op(pre_x,pre_x,pre_x)
    att_x = att_x.contiguous().view(b,n,c,w,h)
    return att_x,relations
