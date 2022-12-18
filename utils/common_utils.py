def feature_digit(a, th, x) -> int:
    """
    implement the signal function: a sgn(x-b) + (1-a)sgn(b-x)  if -1, then 0
    :param a: input a
    :param th: threshold
    :param x: input x
    :return: sgn result
    """
    # a sgn(x-b) + (1-a)sgn(b-x)  if -1, then 0
    if a:
        if x > th:
            return 1
        else:
            return 0
    else:
        if x <= th:
            return 1
        else:
            return 0


def list_to_int(digit_list) -> int:
    """
    convert the list to int
    :param digit_list: list to convert
    :return: digit number
    """
    # binary list to digits
    num = len(digit_list)
    res = 0
    for i in range(num - 1):
        res = 2 * res + digit_list[num - i - 1] * 2
    res += digit_list[0]
    return res
