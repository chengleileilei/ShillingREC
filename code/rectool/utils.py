def early_stopping(value, best, cur_step,best_info,cur_info, max_step, bigger=True):
    r"""validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    # update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            best_info = cur_info
            # update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            best_info = cur_info

            # update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag,best_info#, update_flag


import os


def fileRename(file_path):
    """对文件重命名，防止覆盖重名文件,同时保持文件后缀命不变
    输入: 文件路径
    输出: 新文件路径
    """
    file_path = os.path.abspath(file_path)
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    new_file_path = file_path
    i = 0
    while os.path.isfile(new_file_path):
        i += 1
        prefix_name, suffix = getNameAndSuffix(base_name)
        if suffix == '':
            new_file_path = file_path + '(' + str(i) + ')'
        else:
            new_file_path = os.path.join(
                dir_name, prefix_name + '(' + str(i) + ').' + suffix)
    return new_file_path


def getNameAndSuffix(basename):
    """获取文件后缀和前缀名"""
    l = basename.split('.')
    if (len(l) == 1):
        return l[0], ''
    else:
        suffix = l[-1]
        if suffix == '':
            # basename全为'.'则直接返回
            return basename, ''
        else:
            prefix_name = ''
            for x in l[:-1]:
                prefix_name += x + "."
            return prefix_name[:-1], suffix
