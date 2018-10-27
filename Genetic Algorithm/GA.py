# coding: utf-8
"""
基础遗传算法
1.初始化种群
2.对种群select
3.对种群crossover
4.循环2,3
"""
import numpy as np
import matplotlib.pyplot as plt
import time

class GAB:
    def __init__(self ,pop_size=100 ,dna_size=10,n_generation=200,cross_rate=0.8,mutation_rate=0.003,x_bound=[0,5]):
        self. pop_size =pop_size
        self. dna_size =dna_size
        self. n_generation =n_generation
        self. cross_rate =cross_rate
        self. mutation_rate =mutation_rate
        self. x_bound =x_bound
        # 初始化种群
        self. pop =np.random.randint(2 ,size=(self.pop_size ,self.dna_size))

    """
    主循环部分
    2.对种群select
    3.对种群crossover
    4.循环2,3
    """
    @property
    def training(self):
        plt.ion()       # something about plotting
        x = np.linspace(*self.x_bound)
        plt.plot(x, self.F(x))
        for genneration in range(self.n_generation):
            F_values = self.F(self.translateDNA(self.pop))
            try:
                sca.remove()
            except:
                print("error")
                time.sleep(6)
            sca = plt.scatter(self.translateDNA(self.pop), F_values, s=200, lw=0, c='red', alpha=0.5)
            plt.pause(0.05)
            pop_better=self.select( self.pop)
            pop_next=self.crossover ( pop_better)
            pop_next=self.mutation( pop_next)
            self.pop[:]=pop_next[:]
        plt.ioff(); plt.show()
        return self.pop

    """
    进行自然选择
    1.获取适应度
    2.根据适应度选择个体集合
    """

    def select(self,pop):
        fitness=self.get_fitness(pop)
        better_index=np.random.choice(range( len (pop)),size=len(pop),replace=True ,p=fitness**2/ (fitness**2).sum())
        print ( "better index:")
        print(better_index)
        return pop[better_index]

    """
    获取适应度

    1.减去方程中的最小值,确保适应度都是正数,随后我们需要用这个适应度来作为select种群的概率

    """
    def get_fitness(self,pop):
        F_value= self.F(self.translateDNA( pop))
        return F_value + 1e-3 - np.min(F_value)

    """
    一条自定义的方程,用于反映适应度和画图
    """

    def F(self,x): return np.sin(10*x)*x + np.cos(2*x)*x

    """
    1.把dna当成二进制数
    2.将2进制数转化为10进制,并限制在0到self.x_bound[1]区间内
    """

    def translateDNA(self,pop): return pop.dot(2 ** np. arange(
self.dna_size)[::-1]) / float(2**self.dna_size-1) * self. x_bound [1]

    """
    1.在经过自然选择的种群里
    2.一个个体father 与 种群中随机选出一个个体mother进行随机交换部分dna
    """

    def crossover(self,pop_better):
        pop= pop_better.copy()
        for i in range(len(pop_better)):
            # 父本
            father=pop[i]
            # 母本
            mother_index=np.random.choice( range(len(pop_better)),size=1)
            mother=pop_better[ mother_index][0]
            # 交叉配对的点
            cross_points=np.random.randint(0 , 2,size=len(father), dtype=np.bool)
            # 子代
            father[cross_points]=mother[ cross_points ]
        return pop



    def mutation(self,pop):
        for i in range(len(pop)):
            child=pop[i]
            for j in range(len(child)):
                if np.random.rand()<self. mutation_rate:
                    child[j]=1 if child[j]==0 else 0
        return pop

gab=GAB(100)
print(gab.pop)
print(gab.get_fitness(gab.pop))
print(gab.crossover(gab.select(gab.pop)))

gab.training
