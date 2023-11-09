import tensorflow as tf


def noise_shape(tensor, shape):
    output_shape = list()
    tensor_shape = tf.shape(tensor)
    for i, noise_dim in enumerate(shape):
        if noise_dim is None:
            output_shape.append(tensor_shape[i])
        else:
            output_shape.append(noise_dim)
    return output_shape


def bilinear(x1, x2, W):
    """Bilinear transformation.

    :param x1: tensor with shape (bxn1xd1)
    :param x2: tensor with shape (bxn2xd2)
    :param W: tensor with shape (dx...xd1xd2)
    :return: tensor with shape (bxn1xdx...xn2)
    """
    temp = tf.tensordot(x1, W, axes=[[-1], [-2]])
    temp_shape = tf.shape(temp)
    temp = tf.reshape(temp, (temp_shape[0], -1, temp_shape[-1]))
    result = tf.matmul(temp, x2, transpose_b=True)
    output_shape = [temp_shape[0]] + tf.unstack(temp_shape[1:-1]) + [tf.shape(result)[-1]]
    result = tf.reshape(result, output_shape)
    return result
