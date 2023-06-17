import numpy as np
from typing import Tuple
import cProfile

def avg(o : np.ndarray, window_size : int = None):
    return np.mean(np.lib.stride_tricks.sliding_window_view(o, (o.shape[0], window_size if window_size is not None else len(o))), axis=3)

# TODO : make it work on multiple variables at the same time
def split_gini(array : np.ndarray, target_classes: np.ndarray) -> Tuple[int, float]:
    # count the number of correct call for each class 
    # 2 5 3 9 => 2 3 5 9
    # GI (2.5) -> ((p1 * (1 - p1) + p2 * (1 - p2)) * (num_sample == 1)) == 0 + (((p1 = 1/3) * (1 - (p1 == 1/3)) + (p2 == 2/3) * (1 - (p2 == 2/3))) * (num_sample == 3)) == 0.44 -> 0
    # GI (4) -> GI(< 4) * (num_sample_inf / num_sample) + GI (> 4) * (num_sample_sup / num_sample)
    # for each element in avg(np.sort(x), 2) -> apply :
    # where a is the list of class splited according to the array order list 
    # _, p = np.unique(a) / len(a) -> counts of all classes appearance and proba over number of outcomes
    # GI = 1 - sum(p ** 2)
    ar = np.argsort(array)
    #print(ar)
    data = np.take_along_axis(np.broadcast_to(target_classes, array.shape), ar, axis=1)
    #print(data)
    sorted = np.sort(array)
    interp = avg(sorted, 2)
    min_vals = np.ones(array.shape[0]) - 0.5
    min_indexes = np.zeros(array.shape[0], dtype=int)
    a = len(interp[0])
    #print(a)
    b = len(array[0])
    #print(b)
    for i in range(a):
        split = np.split(data, [i + 1], axis=1)
        #print(split)
        #print(len(split[0][0]))
        #print(split[0])
        #print(split[1])
        p = lambda x, tot_size, split_size : np.sum((np.unique(x, return_counts=True)[1] / split_size) ** 2) * split_size / tot_size
        #print(np.apply_along_axis(f, 1, split[0], b, len(split[0][0])))
        p_l, p_r = (np.apply_along_axis(p, 1, split[0], b, len(split[0][0])), np.apply_along_axis(p, 1, split[1], b, len(split[1][0])))
        #print(p_l, p_r)
        #res = (1 - np.sum(p_l ** 2)) * (len(split[0][0]) / b) + (1 - np.sum(p_r ** 2)) * (len(split[1][0]) / b)
        res = 1 - np.add(p_l, p_r)
        #new = np.where(res < min_vals, i, min_indexes)
        #cond = np.squeeze(np.bitwise_and(min_vals > res, (np.take_along_axis(interp[0], np.expand_dims(min_indexes, 0), 0) < np.take_along_axis(interp[0], np.expand_dims(new, 0), 0))), 0)
        cond = np.bitwise_and(np.take_along_axis(interp, np.broadcast_to(np.array(i), (1, 1, array.shape[0])), axis=1) > sorted[0:, i], min_vals > res)
        #mi_tmp = np.where(res < min_vals, i, min_indexes)
        min_indexes = np.where(cond, i, min_indexes)
        min_vals = np.where(cond, res, min_vals)
        #if np.min(res) < min:
        #    min_index = i
        #    min = res 
    return (np.take_along_axis(interp, min_indexes, 1).reshape(-1), min_vals.reshape(-1))


# TODO : make it work on multiple variables at the same time
def split_gini_childs(array : np.ndarray, target_classes: np.ndarray) -> Tuple[int, float]:
    # count the number of correct call for each class 
    # 2 5 3 9 => 2 3 5 9
    # GI (2.5) -> ((p1 * (1 - p1) + p2 * (1 - p2)) * (num_sample == 1)) == 0 + (((p1 = 1/3) * (1 - (p1 == 1/3)) + (p2 == 2/3) * (1 - (p2 == 2/3))) * (num_sample == 3)) == 0.44 -> 0
    # GI (4) -> GI(< 4) * (num_sample_inf / num_sample) + GI (> 4) * (num_sample_sup / num_sample)
    # for each element in avg(np.sort(x), 2) -> apply :
    # where a is the list of class splited according to the array order list 
    # _, p = np.unique(a) / len(a) -> counts of all classes appearance and proba over number of outcomes
    # GI = 1 - sum(p ** 2)
    ar = np.argsort(array)
    #print(ar)
    data = np.take_along_axis(np.broadcast_to(target_classes, array.shape), ar, axis=1)
    #print(data)
    sorted = np.sort(array)
    interp = avg(sorted, 2)
    as0 = array.shape[0]
    min_vals = np.ones(array.shape[0]) - 0.5
    min_indexes = np.zeros(array.shape[0], dtype=int)
    a = len(interp[0])
    #print(a)
    b = len(array[0])
    to = np.zeros((a, 2, as0))
    #print(b)
    for i in range(a):
        split = np.split(data, [i + 1], axis=1)
        #print(split)
        #print(len(split[0][0]))
        #print(split[0])
        #print(split[1])
        p = lambda x, split_size : np.sum((np.unique(x, return_counts=True)[1] / split_size) ** 2)
        #print(np.apply_along_axis(f, 1, split[0], b, len(split[0][0])))
        p_l, p_r = (np.apply_along_axis(p, 1, split[0], len(split[0][0])), np.apply_along_axis(p, 1, split[1], len(split[1][0])))
        #print(p_l, p_r)
        #res = (1 - np.sum(p_l ** 2)) * (len(split[0][0]) / b) + (1 - np.sum(p_r ** 2)) * (len(split[1][0]) / b)
        to[i] = np.array([1 - p_l, 1 - p_r])
        res = 1 - np.add(p_l * (len(split[0][0]) / b), p_r * (len(split[1][0]) / b))
        #print("left gini", 1 - np.apply_along_axis((lambda x, tot_size, split_size : np.sum((np.unique(x, return_counts=True)[1] / split_size) ** 2)), 1, split[0], b, len(split[0][0])))
        #new = np.where(res < min_vals, i, min_indexes)
        #cond = np.squeeze(np.bitwise_and(min_vals > res, (np.take_along_axis(interp[0], np.expand_dims(min_indexes, 0), 0) < np.take_along_axis(interp[0], np.expand_dims(new, 0), 0))), 0)
        cond = np.bitwise_and(np.take_along_axis(interp, np.broadcast_to(np.array(i), (1, 1, array.shape[0])), axis=1) > sorted[0:, i], min_vals > res)
        #mi_tmp = np.where(res < min_vals, i, min_indexes)
        min_indexes = np.where(cond, i, min_indexes)
        min_vals = np.where(cond, res, min_vals)
        print(np.take_along_axis(interp, np.broadcast_to(np.array(i), (1, 1, array.shape[0])), axis=1))
        #if np.min(res) < min:
        #    min_index = i
        #    min = res 
    return (np.take_along_axis(interp, min_indexes, 1).reshape(-1), min_vals.reshape(-1), np.take_along_axis(to, min_indexes, 0).T.reshape(-1, 2))


#x = np.array([False, False, False, True, True, True, True])
#y = np.array([1, 1, 0, 1, 1, 1, 0])
#print(num_split_gini(x, y))

x = np.array([[True, True, False, False, True, True, False], [True, False, True, True, True, False, False], [7, 12, 18, 35, 38, 50, 83]])
y = np.array([False, False, True, True, True, False, False])

#print(cProfile.run("split_gini(x, y)"))
print(split_gini_childs(x, y))
#print(split_gini(x, y))
thresholds, gini, child_nodes_tuples = split_gini_childs(x, y)
min_index = gini.argmin()

cond = x[min_index] > thresholds[min_index]
r = np.delete(x, np.where(cond)[0], 1)
l = np.delete(x, np.where(1 - cond)[0], 1)
y_r = y[np.where(1 - cond)[0]]
y_l = y[np.where(cond)[0]]

x = np.array([[7, 12, 18, 35, 38, 50, 83], [7, 12, 19, 3, 38, -5, 83], [0, 0, 0, 0, 0, 0, 0]])
y = np.array([0, 0, 1, 1, 1, 0, 0])

#print(cProfile.run("split_gini(x, y)"))
print(split_gini_childs(x, y))
#print(split_gini(x, y))