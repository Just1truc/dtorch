""" imports """
import dtorch as dt
from typing import Optional, Tuple
from multiprocessing import Process, Queue
from abc import ABC, abstractmethod
import numpy as np
import pickle
import copy
from dtorch.nn import JPretrainedModel

class DTorchTreeModel(ABC):

    """ properties """

    @property
    def accuracy(self):
        raise NotImplementedError

    """ methods """

    @abstractmethod
    def fit(x : dt.jtensors.JTensors, y : dt.jtensors.JTensors) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def predict(x : dt.jtensors.JTensors):
        raise NotImplementedError
    
    @abstractmethod
    def save(path : str):
        raise NotImplementedError
    
    @abstractmethod
    def load(path : str):
        raise NotImplementedError


class Node:
    
    def __init__(self,
                 threshold : int,
                 gini : float,
                 feature : int,
                 left = None,
                 right = None,
                 left_pred : int = None,
                 right_pred : int = None) -> None:
        self.feature : int = feature
        self.threshold : int = threshold
        self.gini : float = gini
        self.left = left
        self.right = right
        self.left_pred = left_pred
        self.right_pred = right_pred


    def dump(self):
        res = copy.copy(self)
        res.left = pickle.dumps(res.left and res.left.dump(), protocol=-1)
        res.right = pickle.dumps(res.right and res.right.dump(), protocol=-1)
        return res
    
    def load(self):
        self.left = pickle.loads(self.left)
        self.right = pickle.loads(self.right)
        self.left = self.left and self.left.load()
        self.right = self.right and self.right.load()
        return self
    

    def __reccursive_str__(self, depth : int) -> str:
        left = ""
        right = ""
        if (self.left is not None):
            left = "|\n" + "|\t" * depth + "+-- " + self.left.__reccursive_str__(depth + 1)
        else:
            left = "|\n" + "|\t" * depth + "+-- " + str(self.left_pred) + "\n"
        if (self.right is not None):
            right = "|\n" + "|\t" * depth + "+-- " + self.right.__reccursive_str__(depth + 1)
        else:
            right = "|\n" + "|\t" * depth + "+-- " + str(self.right_pred) + "\n"
        return f"Node (feature = {self.feature}, threshold <= {self.threshold}, gini = {self.gini})\n" + "|\t" * depth + left + "|\t" * depth + right


    def __str__(self) -> str:
        return self.__reccursive_str__(0)
    

    def __repr__(self) -> str:
        return self.__str__()


class DecisionTree(DTorchTreeModel):

    def __init__(self,
                 criterion : str = 'gini',
                 max_depth : Optional[int] = None,
                 multiprocesses_threshold : Optional[int] = 0.4
                 ) -> None:
        """Decision tree very basic with almost no control on the growing tree

        Args:
            criterion (str, optional): 'gini', 'entropy' function for split choice. Defaults to 'gini'.
            max_depth (Optional[int], optional): max depth of the tree. Defaults to None.
            multiprocesses (Optional[int], optional): thresold at which the criterion a new process is created.
        """
        
        assert (criterion in ['gini', 'entropy'])
        assert (max_depth is None or max_depth >= 1)

        self.__criterion : str = criterion
        self.__max_depth : int = max_depth

        self.__head : Node | None = None
        self.__process_threshold : int = multiprocesses_threshold
        self.__accuracy : float = 0.0

    
    @property
    def accuracy(self):
        return self.__accuracy


    def avg(self, o : np.ndarray, window_size : int = None):
        return np.mean(np.lib.stride_tricks.sliding_window_view(o, (o.shape[0], window_size if window_size is not None else len(o))), axis=3)


    # TODO : make it work on multiple variables at the same time
    def split_gini(self, array : np.ndarray, target_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        interp = self.avg(sorted, 2)
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
            #new = np.where(res < min_vals, i, min_indexes)
            #cond = np.squeeze(np.bitwise_and(min_vals > res, (np.take_along_axis(interp[0], np.expand_dims(min_indexes, 0), 0) < np.take_along_axis(interp[0], np.expand_dims(new, 0), 0))), 0)
            cond = np.bitwise_and(np.take_along_axis(interp, np.broadcast_to(np.array(i), (1, 1, array.shape[0])), axis=1) > sorted[0:, i], min_vals > res)
            #mi_tmp = np.where(res < min_vals, i, min_indexes)
            min_indexes = np.where(cond, i, min_indexes)
            min_vals = np.where(cond, res, min_vals)
            #if np.min(res) < min:
            #    min_index = i
            #    min = res 
        return (np.take_along_axis(interp, min_indexes, 1).reshape(-1), min_vals.reshape(-1), np.take_along_axis(to, min_indexes, 0).T.reshape(-1, 2))
    
    # if depth == max_depth : stop else
    # calculate less cost effective gini impurity parameter for the given x and y
    # split the data according to the threshold
    # check if impurity of child node is superior to process threshold value -> create new process.

    def calculate(self,
                  x : dt.jtensors.JTensors,
                  y : dt.jtensors.JTensors,
                  cur_depth : int,
                  q : Optional[Queue] = None) -> Optional[Node]:
        
        #print("y", y)

        thresholds, gini, child_nodes_tuples = u = self.split_gini(x, y)
        #print(u)
        min_index = gini.argmin()
        new_node = Node(thresholds[min_index], gini.min(), min_index + 1, None, None)

        if self.__max_depth and cur_depth > self.__max_depth:
            return new_node
        
        cond = x[min_index] <= thresholds[min_index]
        a = np.where(cond)[0]
        r = np.delete(x, a, 1)
        l = np.delete(x, np.where(1 - cond)[0], 1)
        y_r = y[np.where(1 - cond)[0]]
        y_l = y[np.where(cond)[0]]
        pl = None
        ql = None
        pr = None
        qr = None

        # multiprocess
        if child_nodes_tuples[min_index][0] > self.__process_threshold and child_nodes_tuples[min_index][0] > 0:
            #print("Starting process left...", child_nodes_tuples[min_index][0])
            ql = Queue()
            pl = Process(target=self.calculate, args=(l, y_l, cur_depth + 1, ql))
            pl.start()
        elif child_nodes_tuples[min_index][0] > 0:
            new_node.left = self.calculate(l, y_l, cur_depth + 1)
        else:
            u, c = np.unique(y_l, return_counts=True)
            new_node.left_pred = u[np.argmax(c)]
        
        if child_nodes_tuples[min_index][1] > self.__process_threshold and child_nodes_tuples[min_index][1] > 0:
            #print("Starting process right...", child_nodes_tuples[min_index][1])
            qr = Queue()
            pr = Process(target=self.calculate, args=(r, y_r, cur_depth + 1, qr))
            pr.start()
        elif child_nodes_tuples[min_index][1] > 0:
            new_node.right = self.calculate(r, y_r, cur_depth + 1)
        else:
            u, c = np.unique(y_r, return_counts=True)
            new_node.right_pred = u[np.argmax(c)]

        if pr != None:
            pr.join()
            new_node.right = pickle.loads(qr.get()).load()
        if pl != None:
            pl.join()
            new_node.left = pickle.loads(ql.get()).load()
        
        # calculate sub node (can't put in the q right away because sub elements may be ignored)
        if (q != None):
            q.put(pickle.dumps(new_node.dump(), protocol=-1))
        else:
            return new_node


    def fit(self, x : dt.jtensors.JTensors, y : dt.jtensors.JTensors):

        assert (x.shape[-1] == y.shape[0]), "X must be of shape (samples, features) and Y (samples,)"

        self.__head = self.calculate(x, y, 0, None)


    def predict(self, x: dt.jtensors.JTensors):
        
        node : Node = self.__head
        cclass = 0
        while node != None:
            if x[node.feature - 1] > node.threshold:
                cclass = node.right_pred
                node = node.right
            else:
                cclass = node.left_pred
                node = node.left

        return cclass


    def save(self, path : str):
        with open(path, "wb") as bf:
            bf.write(pickle.dumps(self.__head.dump()))


    def load(self, path : str):
        with open(path, "rb") as bf:
            self.__head = pickle.loads(bf.read())
            self.__head = self.__head and self.__head.load()


    def __str__(self) -> str:
        return f"DecisionTree (criterion = {self.__criterion})\n" + str(self.__head)


    def __repr__(self) -> str:
        return self.__str__()



class IrisDecisionTree(JPretrainedModel):

    def __init__(self, root: str = './models') -> None:
        super().__init__('iris_model', 'dtorchtree', root)

        self.__model : DTorchTreeModel = DecisionTree()
        self.__model.load(self.model_path)


    def predict(self, x : dt.jtensors.JTensors):
        return self.__model.predict(x)
    

    def __str__(self) -> str:
        return self.__model.__str__()
    
    
    def __repr__(self) -> str:
        return self.__str__()