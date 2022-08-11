import os 

path = 'dataset'
train_path  = path + '/train/'
test_path  = path + '/test/'
eval_path = path + '/val/'

def count_images(path):
    l = os.listdir(path)
    m = 0
    for i in l:
        m += len(os.listdir(path + i))

    return m 

n_train, n_val, n_test = count_images(train_path), count_images(eval_path), count_images(test_path)
print(' train count: ', n_train)
print(' val count: ', n_val)
print(' test count: ', n_test) 

