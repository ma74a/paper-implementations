import splitfolders


def split_data():

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio("data_dir", output="splited_dir",
        seed=1337, ratio=(.8, .2), group_prefix=None, group=None,
        formats=None, move=False, shuffle=True) # default values


