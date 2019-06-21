import math


class Recommender:
    def __init__(self, data, k=1, metrics="pearson", n=3):
        """
        :param data: dictionary type
        :param k: the k value for k nearest neighbor
        :param metrics: distance used
        :param n: the maximum number of recommendations to make
        """
        self.k = k
        self.n = n
        self.metrics = metrics
        if self.metrics == "pearson":
            self.fn = self.pearson
        elif self.metrics == "minkowshi":
            self.fn = self.minkowshi
        if type(data).__name__ == "dict":
            self.data = data

    def pearson(self, series_1, series_2):
        """
        Compute pearson correlation between 2 series    
        """
        total_prod = 0
        total_x = 0
        total_y = 0
        total_x_sqrt = 0
        total_y_sqrt = 0
        n = 0
        for key in series_1:
            if key in series_2:
                n += 1
                x = series_1[key]
                y = series_2[key]
                total_prod += x * y
                total_x += x
                total_y += y
                total_x_sqrt += pow(x, 2)
                total_y_sqrt += pow(y, 2)
        if n == 0:
            return 0
        denominator = math.sqrt(total_x_sqrt - pow(total_x, 2) / n) * math.sqrt(
            total_y_sqrt - pow(total_y, 2) / n
        )
        if denominator == 0:
            return 0
        else:
            return (total_prod - (total_x * total_y) / n) / denominator

    def minkowshi(self, series_1, series_2, r=2):
        """
        Compute the Minkowshi's distance
        """
        distance = 0
        commonCheck = False
        for key in series_1:
            if key in series_2:
                distance += pow(abs(series_1[key] - series_2[key]), r)
                commonCheck = True
        if commonCheck:
            return pow(distance, 1 / r)
        else:
            return -1

    def compute_nearest_neighbor(self, selected_key):
        """
        Creates a sorted list of keys based on their distance to selected key
        """
        distances = []
        for instance in self.data:
            if instance != selected_key:
                distance = self.fn(self.data[selected_key], self.data[instance])
                distances.append((instance, distance))
        distances.sort(key=lambda keyTuple: keyTuple[1], reverse=True)
        return distances

    def recommend(self, selected_key):
        """
        Gives a list of recommendations
        """
        recommendations = {}
        nearest = self.compute_nearest_neighbor(selected_key)
        selected_key_values = self.data[selected_key]
        total_distance = 0.0
        for i in range(self.k):
            total_distance += nearest[i][1]
        for i in range(self.k):
            weight = nearest[i][1] / total_distance
            name = nearest[i][0]
            neighbor_values = self.data[name]
            for key in neighbor_values:
                if not key in selected_key_values:
                    if key not in recommendations:
                        recommendations[key] = neighbor_values[key] * weight
                    else:
                        recommendations[key] = (
                            recommendations[key] + neighbor_values[key] * weight
                        )
        recommendations = list(recommendations.items())
        recommendations.sort(key=lambda keyTuple: keyTuple[1], reverse=True)
        return recommendations[: self.n]


def main():
    users = {
        "Angelica": {
            "Blues Traveler": 3.5,
            "Broken Bells": 2.0,
            "Norah Jones": 4.5,
            "Phoenix": 5.0,
            "Slightly Stoopid": 1.5,
            "The Strokes": 2.5,
            "Vampire Weekend": 2.0,
        },
        "Bill": {
            "Blues Traveler": 2.0,
            "Broken Bells": 3.5,
            "Deadmau5": 4.0,
            "Phoenix": 2.0,
            "Slightly Stoopid": 3.5,
            "Vampire Weekend": 3.0,
        },
        "Chan": {
            "Blues Traveler": 5.0,
            "Broken Bells": 1.0,
            "Deadmau5": 1.0,
            "Norah Jones": 3.0,
            "Phoenix": 5,
            "Slightly Stoopid": 1.0,
        },
        "Dan": {
            "Blues Traveler": 3.0,
            "Broken Bells": 4.0,
            "Deadmau5": 4.5,
            "Phoenix": 3.0,
            "Slightly Stoopid": 4.5,
            "The Strokes": 4.0,
            "Vampire Weekend": 2.0,
        },
        "Hailey": {
            "Broken Bells": 4.0,
            "Deadmau5": 1.0,
            "Norah Jones": 4.0,
            "The Strokes": 4.0,
            "Vampire Weekend": 1.0,
        },
        "Jordyn": {
            "Broken Bells": 4.5,
            "Deadmau5": 4.0,
            "Norah Jones": 5.0,
            "Phoenix": 5.0,
            "Slightly Stoopid": 4.5,
            "The Strokes": 4.0,
            "Vampire Weekend": 4.0,
        },
        "Sam": {
            "Blues Traveler": 5.0,
            "Broken Bells": 2.0,
            "Norah Jones": 3.0,
            "Phoenix": 5.0,
            "Slightly Stoopid": 4.0,
            "The Strokes": 5.0,
        },
        "Veronica": {
            "Blues Traveler": 3.0,
            "Norah Jones": 5.0,
            "Phoenix": 4.0,
            "Slightly Stoopid": 2.5,
            "The Strokes": 3.0,
        },
    }
    r = Recommender(users)
    r_euclidean = Recommender(users, metrics="minkowshi")
    print(r.recommend("Jordyn"))
    print(r.recommend("Hailey"))
    print(r.recommend("Angelica"))
    print(r.recommend("Bill"))
    print("\n")
    print(r_euclidean.recommend("Jordyn"))
    print(r_euclidean.recommend("Hailey"))
    print(r_euclidean.recommend("Angelica"))
    print(r_euclidean.recommend("Bill"))
    # print(r.pearson(users["Jordyn"], users["Hailey"]))


if __name__ == "__main__":
    main()
