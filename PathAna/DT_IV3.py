from collections import namedtuple,defaultdict
import numpy as np

TreeNode = namedtuple("TreeNode", 'feature iv branch label')

class DecisionTreeClassifier(object):
    def __init__(self):
        pass

    @staticmethod
    def devide_X(X_, feature):
        """切分集合
        :param X_: 被切分的集合, shape=[ni_samples, ni_features + 1]
        :param feature: 切分变量
        :param val: 切分变量的值
        :return: 数据集合list
        """
        X_list=[]
        feature_unique_val=np.unique(X_[:, feature])
        for _ in feature_unique_val:
            X_list.append(X_[X_[:, feature] == _])
        return X_list,feature_unique_val
    @staticmethod
    def iv_index(cluster, attr_index):
        """
        :param cluster: 训练集的一个子集
        :param attr_index:  特征编号（第N个特征）
        :return: 第N个特征的的IV
        """
        yi_dict = defaultdict(int)  # 每个特征，每种值中label =1的数量
        ni_dict = defaultdict(int)  # 每个特征，每种值 label =0的数量
        label_t_dict = defaultdict(int)  # label=1或0的数量
        for line in cluster:
            yi_dict[line[attr_index]] += line[-1]
            ni_dict[line[attr_index]] += (1 - line[-1])
            label_t_dict[line[-1]] += 1
        yi_arr = np.array(list(yi_dict.values()))
        ni_arr = np.array(list(ni_dict.values()))

        pyi = yi_arr / label_t_dict[1]
        pni = ni_arr / label_t_dict[0]
        pyi[pyi == 0] = 1
        pni[pni == 0] = 1  # Laplace平滑，令概率不为0

        pyi_div_pni = pyi / pni
        iv_i_arr = ((pyi) - (pni)) * np.log(pyi_div_pni)
        #         iv_i_arr[iv_i_arr > 0.8] = 0

        return sum(iv_i_arr)

    @staticmethod
    def get_best_iv_index(cluster, attr_indexs):
        '''
        :param cluster: 给定数据集
        :param attr_indexs: 给定的可供切分的特征编号的集合
        :return: 最佳切分特征，该特征的iv得分
        '''
        p = {}
        for attr_index in attr_indexs:
            p[attr_index] = DecisionTreeClassifier.iv_index(cluster, attr_index)
        attr_index = max(p, key=lambda x: p.get(x))#这里返回的是value最大的key值
        attr = p[attr_index]
        return attr_index, attr

    def build(self, X_, features):
        """建树
        :param X_: 候选集 shape=[ni_samples, n_features + 1]
        :param features: 候选特征集
        :param depth: 当前深度
        :return: 结点
        """

        if np.unique(X_[:, -1]).shape[0] == 1:
            return TreeNode(None, None, None, X_[0, -1])
        if features.shape[0] == 0 :
            classes, classes_count = np.unique(X_[:, -1], return_counts=True)
            return TreeNode(None, None, None, classes[np.argmax(classes_count)])
        feature_index, iv = DecisionTreeClassifier.get_best_iv_index(X_, features)
        new_features = features[features != feature_index]
        del features

        X_list,feature_unique_val = DecisionTreeClassifier.devide_X(X_, feature_index)
        branch_dict={}
        for fea_val,_ in zip(feature_unique_val,X_list):
            branch_temp = self.build(_, new_features)
            branch_dict[fea_val]=(branch_temp)

        return TreeNode(feature_index, iv, branch_dict, None)

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features]
        :param y: shape = [n_samples]
        :return: self
        """
        features = np.arange(X.shape[1])
        X_ = np.c_[X, y]
        self.root = self.build(X_, features)
        return self

    def predict_one(self, x):
        p = self.root
        while p.label is None:
            p = p.branch_dict[x[p.feature]]
        return p.label

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features]
        :return: shape = [n_samples]
        """
        return np.array([self.predict_one(x) for x in X])

train=np.array([['ex1','no2','co4'],
                ['ex2','no2','co1'],
                ['ex2','no3','co3'],
                ['ex2','no1','co1'],
                ['ex1','no1','co4'],
                ['ex1','no1','co1'],
                ['ex3','no3','co2'],
                ['ex3', 'no3', 'co1'],
                ['ex1', 'no3', 'co2'],
                ['ex3', 'no1', 'co3'],
                ['ex1', 'no1', 'co4'],
                ['ex3', 'no1', 'co4'],
                ])
label_array=np.array([1,0,1,0,1,0,0,0,1,1,1,0])
print(train.shape,label_array.shape)
print(np.c_[train,label_array])
exit()
clf_iv=DecisionTreeClassifier()
clf_iv.fit(train,label_array)

