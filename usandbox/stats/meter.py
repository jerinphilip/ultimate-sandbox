import pandas

class Meter:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.data = []

    def report(self, value):
        self.data.append(value)
        self.sum += value
        self.count += 1
        
    def avg(self):
        val = float('inf') if not self.count else self.sum/self.count
        return val

    def __str__(self):
        return '{:.2f}'.format(self.avg)

    def export(self):
        xs = range(1, len(self.data))
        ys = self.data
        d = {"t": xs, "y": ys}
        return pandas.DataFrame(d)
    
    
