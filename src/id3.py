import numpy as np

epsilon = 10e-7 # menghidari log 0

class DecisionTree:
    """Implementasi algoritma Decision Tree dalam Python"""
    def __init__(self):
        self._X = None
        self._y = None
        self._tree = dict()
        
    def _entropy(self, att: np.ndarray):
        _, counts = np.unique(att, return_counts = True)
        frac = counts / sum(counts)

        return sum(-frac * np.log2(frac))

    def _entropy_given_y(self, att: np.ndarray, y: np.ndarray):
        """
            Entropi dengan kondisi y sebagai target
        
            params:
                att: Kolom atribut dengan dimensi (m,) dengan m adalah banyaknya data
                y: Kolom target dengan dimensi (m,) yang berkorespondensi dengan att
        """

        categories_att = np.unique(att)
        categories_y = np.unique(y)
        total = len(att)

        ent = 0
        for cat in categories_att:
            idx = (att == cat)
            cat_total = sum(idx)
            tmp = 0
            for cat_y in categories_y:
                count = (y[idx] == cat_y)
                frac = sum(count) / cat_total
                tmp += - frac * np.log2(frac + epsilon)
            ent += sum(idx) / total * tmp

        return ent

    def _calculate_all_entropy(self, X: np.ndarray, y: np.ndarray, att_index: list):
        """
            Menghitung entropi untuk semua atribut yang disertakan pada att_index

            Params
                X: data dengan dimensi (m, n) dengan m adalah jumlah baris dan n adalah jumlah fitur
                att_index: indeks dari atribut yang akan dihitung, berdimensi (i,) dengan i adalah banyak fitur yang dipilih
                y: kolom target dengan dimensi (m,)
        """
        results = np.zeros(len(att_index))

        for i in range(len(att_index)):
            ent = self._entropy_given_y(X.T[i], y)
            results[i] = ent

        return results

    def _calculate_information_gain(self, X: np.ndarray, y: np.ndarray, att_index: list):
        """
            Menghitung information gain untuk setiap atribut dengan indeks att_index

            Params
                X: data dengan dimensi (m, n) dengan m sebagai jumlah row dan n adalah jumlah fitur
                y: label
                att_index: indeks dari atribut yang akan dihitung, berdimensi (i,) dengan i adalah banyak fitur yang dipilih
        """

        return self._entropy(y) - self._calculate_all_entropy(X, y, att_index)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray):
        """
            Membangun decision tree

            Params
                X: data dengan dimensi (m, n) dengan m sebagai jumlah row dan n adalah jumlah fitur
                y: label
        """
        columns = list(range(X.shape[1]))

        highest_gain = np.argmax(self._calculate_information_gain(X, y, columns))
        att_categories = np.unique(X.T[highest_gain])

        tree = {}
        tree[highest_gain] = {}

        for cat in att_categories:
            indices = X.T[highest_gain] == cat

            cat_y = np.unique(y[indices])

            if len(cat_y) == 1:
                # pure class
                tree[highest_gain][cat] = cat_y[0]
            else:
                tree[highest_gain][cat] = self._build_tree(X[indices], y[indices])

        return tree
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Membentuk decision tree dari X dan y yang diberikan
            
            Params
                X: Matriks dengan dimensi (m, n) dengan m sebagai jumlah data dan n sebagai jumlah fitur.
                   Jumlah fitur harus dalam tipe kategorikal.
                y: Array berdimensi (m,) sebagai target label.
        """
        self._X = X
        self._y = y
        self._tree = self._build_tree(self._X, self._y)
        
        return self

    def predict(self, X: np.ndarray):
        """
            Melakukan prediksi pada tree yang telah dibuat

            X: array berukuran (m, n)
        """

        y = []
        
        if self._X is None or self._y is None or self._tree == {}:
            raise Exception("Model belum di-fit")
            
        for row in X:
            result = self._tree
            while type(result) is dict:
                attribute_index = list(result.keys())[0]
                inner_tree = result[attribute_index]
                result = inner_tree.get(row[attribute_index])
            y.append(result)
        return y