import numpy as np
from copy import deepcopy
import time
import math
import scipy.stats


class BinaryProblem:
    """
    An abstract class for an individual in binary representation
    """
    def __init__(self, n, val=None, rng=np.random.default_rng()):
        if val is not None:
            assert isinstance(val, bool)
            self.data = np.array([val] * n)
        else:
            self.data = rng.choice([True,False], size=n) 
        self.n = n
        self.fitness = self.eval()
        

    def is_optimal(self):
        pass


    def get_optimal(self):
        pass


    def eval(self):
        pass        


    def get_fitness_after_flipping(self, locs):
        """
        Calculate the change in fitness after flipping the bits at positions locs

        Parameters
        -----------
            locs: 1d-array
                positions where bits are flipped

        Returns: int
        -----------
            objective after flipping
        """
        raise NotImplementedError

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """
        Calculate fitness of the child aftering being crossovered with xprime

        Parameters
        -----------
            xprime: 1d boolean array
                the individual to crossover with
            locs_x: 1d boolean/integer array
                positions where we keep current bits of self
            locs_xprime: : 1d boolean/integer array
                positions where we change to xprime's bits

        Returns: fitness of the new individual after crossover
        -----------            
        """
        raise NotImplementedError

    def flip(self, locs):
        """
        flip the bits at position indicated by locs

        Parameters
        -----------
            locs: 1d-array
                positions where bits are flipped

        Returns: the new individual after the flip
        """
        child = deepcopy(self)
        child.data[locs] = ~child.data[locs]
        child.eval()
        return child

    def combine(self, xprime, locs_xprime):
        """
        combine (crossover) self and xprime by taking xprime's bits at locs_xprime and self's bits at other positions

        Parameters
        -----------
            xprime: 1d boolean array
                the individual to crossover with
            locs_x: 1d boolean/integer array
                positions where we keep current bits of self
            locs_xprime: : 1d boolean/integer array
                positions where we change to xprime's bits

        Returns: the new individual after the crossover        

        """
        child = deepcopy(self)
        child.data[locs_xprime] = xprime.data[locs_xprime]
        child.eval()
        return child

    def mutate(self, p, n_childs, rng=np.random.default_rng()):
        """
        Draw l ~ binomial(n, p), l>0
        Generate n_childs children by flipping exactly l bits
        Return: the best child (maximum fitness), its fitness and number of evaluations used        
        """
        assert p>=0

        if p==0:
            return self, self.fitness, 0

        p_distribution = [0] + [scipy.stats.binom.pmf(k, self.n, p) / (1-(1-p)**self.n) for k in range(1, self.n+1)]        
        l = rng.choice(range(self.n + 1), p=p_distribution)               
        
        best_obj = -1
        best_locs = None
        for i in range(n_childs):
            locs = rng.choice(self.n, size=l, replace=False)        
            obj = self.get_fitness_after_flipping(locs)
            if obj > best_obj:
                best_locs = locs
                best_obj = obj                       

        best_child = self.flip(best_locs)                

        return best_child, best_child.fitness, n_childs

    def mutate_rls(self, l, rng=np.random.default_rng()):
        """
        a child by flipping exactly l bits
        Return: child, its fitness        
        """
        assert l>=0

        if l==0:
            return self, self.fitness, 0

        locs = rng.choice(self.n, size=l, replace=False) 
        child = self.flip(locs)

        return child, child.fitness       

    def crossover(self, xprime, p, n_childs, 
                    include_xprime=True, count_different_inds_only=True,
                    rng=np.random.default_rng()):
        """
        Crossover operator:
            for each bit, taking value from x with probability p and from self with probability 1-p
        Arguments:
            x: the individual to crossover with
            p (float): in [0,1]                                                
        """
        assert p <= 1
        
        if p == 0:
            if include_xprime:
                return xprime, xprime.fitness, 0
            else:
                return self, self.fitness, 0            

        if include_xprime:
            best_obj = xprime.fitness
        else:
            best_obj = -1            
        best_locs = None

        n_evals = 0
        ls = rng.binomial(self.n, p, size=n_childs)        
        for l in ls:                   
            locs_xprime = rng.choice(self.n, l, replace=False)
            locs_x = np.full(self.n, True)
            locs_x[locs_xprime] = False
            obj = self.get_fitness_after_crossover(xprime, locs_x, locs_xprime) 
                   
            if (obj != self.fitness) and (obj!=xprime.fitness):
                n_evals += 1
            elif (not np.array_equal(xprime.data[locs_xprime], self.data[locs_xprime])) and (not np.array_equal(self.data[locs_x], xprime.data[locs_x])):            
                n_evals += 1            

            if obj > best_obj:
                best_obj = obj
                best_locs = locs_xprime
            
            
        if best_locs is not None:
            child = self.combine(xprime, best_locs)
        else:
            child = xprime

        if not count_different_inds_only:
            n_evals = n_childs

        return child, child.fitness, n_evals


class OneMax(BinaryProblem):
    """
    An individual for OneMax problem
    The aim is to maximise the number of 1 bits
    """

    def eval(self):
        self.fitness = self.data.sum()
        return self.fitness

    def is_optimal(self):
        return self.data.all()

    def get_optimal(self):
        return self.n

    def get_fitness_after_flipping(self, locs):        
        # f(x_new) = f(x) + l - 2 * sum_of_flipped_block
        return self.fitness + len(locs) - 2 * self.data[locs].sum()

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):        
        return self.data[locs_x].sum() + xprime.data[locs_xprime].sum()
        

class LeadingOne(BinaryProblem):    
    """
    An individual for LeadingOne problem
    The aim is to maximise the number of leading (and consecutive) 1 bits in the string
    """

    def eval(self):
        k = self.data.argmin()
        if self.data[k]:
            self.fitness = self.n
        else:
            self.fitness = k
        return self.fitness

    def is_optimal(self):
        return self.data.all()  

    def get_optimal(self):
        return self.n    

    def get_fitness_after_flipping(self, locs):        
        min_loc = locs.min()
        if min_loc < self.fitness:
            return min_loc
        elif min_loc > self.fitness:
            return self.fitness
        else:
            old_fitness = self.fitness
            self.data[locs] = ~self.data[locs]
            new_fitness = self.eval()            
            self.data[locs] = ~self.data[locs]
            self.fitness = old_fitness
            return new_fitness


    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        child = self.combine(xprime, locs_xprime)                
        child.eval()
        return child.fitness
        

def get_default_rng(seed):
    return np.random.default_rng(seed)


def onell_static(n, problem=OneMax, seed=None,
                    lbd1=5, lbd2=60, k=7, c=0.014,                     
                    max_evals = 99999999,
                    count_different_inds_only=True,                    
                    include_xprime_crossover = True               
                ):
    """
    (1+LL)-GA, static version with 4 hyper-parameters as in https://arxiv.org/pdf/1904.04608.pdf

    Arguments:
        lbd1 (int): number of off-springs in the mutation phase
        lbd2 (int): number of off-springs in the crossover phase
        k (int): in the mutation phase, exactly l bits is flipped where l ~ Binomial(size, k/size)
        c (float): in the crossover phase, child bits are taken with probability c
        count_different_inds_only (bool): if True, only count the evaluation of an offspring (in the crossover phase) if it's different from both of its parents            
    Returns:
        best solution, its fitness
    """
    rng = get_default_rng(seed)    
    x = problem(n, rng=rng)
    f_x = x.fitness    
    p = k/n

    total_evals = 1
    while not x.is_optimal():
        xprime, f_xprime, ne1 = x.mutate(p, lbd1, rng)
        y, f_y, ne2 = x.crossover(xprime, c, lbd2, include_xprime_crossover, count_different_inds_only, rng)

        total_evals = total_evals + ne1 + ne2
        if f_x <= f_y:
            x = y
            f_x = f_y

        #print("%d %d" % (x.fitness, total_evals))
        # DEBUG
        #print("%d %d %d %d" % (total_evals, xprime.fitness, y.fitness, x.fitness))
        #quit()

        if total_evals >= max_evals:
            break            

    return x, f_x, total_evals

def onell_dynamic_5params(n, problem=OneMax, seed=None,
                    alpha=0.45, beta=1.6, gamma=1, A=1.16, b=0.7,                    
                    max_evals = 99999999,
                    count_different_inds_only=True,                    
                    include_xprime_crossover = True           
                    
                ):
    """
    (1+LL)-GA, dynamic version with 5 hyper-parameters as in https://arxiv.org/pdf/1904.04608.pdf
    The default hyper-parameter setting here is the best one found in that paper

    Arguments:

    Returns:

    """
    assert A>1 and b<1

    rng = get_default_rng(seed)

    x = problem(n, rng=rng)   
    f_x = x.fitness

    lbd = 1
    min_prob = 1/n
    max_prob = 0.99

    total_evals = 1 # total number of solution evaluations    

    mtimes, ctimes = [], []
    steps = 0
    while not x.is_optimal():
        # mutation phase
        s = time.time()
        p =  np.clip(alpha * lbd / n, min_prob, max_prob)               
        xprime, f_xprime, ne1 = x.mutate(p, round(lbd), rng)  
        mtimes.append(time.time()-s)            
     
        # crossover phase
        s = time.time()
        c = np.clip(gamma / lbd, min_prob, max_prob)        
        n_childs = round(lbd * beta)
        y, f_y, ne2 = x.crossover(xprime, c, n_childs, include_xprime_crossover, count_different_inds_only,  rng)      
        ctimes.append(time.time()-s)     
        
        # update parameters        
        if x.fitness < y.fitness:        
            lbd = max(b*lbd, 1)
        else:
            lbd = min(A*lbd, n-1)

        # selection phase
        old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y

        total_evals = total_evals + ne1 + ne2

        steps += 1
        #print("%d: evals=%d; x=%d; xprime=%d; y=%d; obj=%d; p=%.2f; c=%.2f; lbd=%.2f" % (steps, total_evals, old_f_x,xprime.fitness, y.fitness, x.fitness, p, c, lbd)) 
            
        if total_evals>=max_evals:
            total_evals *= 2
            break

    #print(total_evals)
        
    return x, f_x, total_evals #, mtimes, ctimes

def onell_dynamic_5params_old(n, problem=OneMax, seed=None,
                    alpha=0.45, beta=1.6, gamma=1, A=1.16, b=0.7,                    
                    max_evals = 99999999,
                    count_different_inds_only=True,                    
                    include_xprime_crossover = True           
                ):
    """
    (1+LL)-GA, dynamic version with 5 hyper-parameters as in https://arxiv.org/pdf/1904.04608.pdf
    The default hyper-parameter setting here is the best one found in that paper

    Arguments:

    Returns:

    """
    assert A>1 and b<1

    rng = get_default_rng(seed)

    x = problem(n, rng=rng)   
    f_x = x.fitness

    lbd = 1
    min_prob = 1/n
    max_prob = 0.99

    total_evals = 1 # total number of solution evaluations    

    mtimes, ctimes = [], []
    while not x.is_optimal():
        # mutation phase
        s = time.time()
        p =  np.clip(alpha * lbd / n, min_prob, max_prob)                
        xprime, f_xprime, ne1 = x.mutate_old(p, round(lbd), rng)      
        mtimes.append(time.time()-s)     

        # crossover phase
        s = time.time()
        c = np.clip(gamma / lbd, min_prob, max_prob)        
        n_childs = round(lbd * beta)
        y, f_y, ne2 = x.crossover_old(xprime, c, n_childs, include_xprime_crossover, count_different_inds_only,  rng)          
        ctimes.append(time.time()-s)     
        
        # update parameters                
        if x.fitness < y.fitness:        
            lbd = max(b*lbd, 1)
        else:
            lbd = min(A*lbd, n-1)

        # selection phase
        old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y

        total_evals = total_evals + ne1 + ne2                
            
        if total_evals>=max_evals:
            break  

    #print(total_evals)
        
    return x, f_x, total_evals #, mtimes, ctimes    

def onell_lambda(n, problem=OneMax, seed=None,
                    lbds = None,
                    max_evals = 99999999,
                    count_different_inds_only=True,                    
                    include_xprime_crossover = True,
                ):
    """
    (1+LL)-GA, dynamic version with theoretical results
    lbd = sqrt(n*(n-f(x))), p = lbd/n, c=1/lbd    

    """

    if not lbds:
        lbds = [1] * n
    else: 
        lbds = lbds.copy()
    assert len(lbds) == n

    lbds.append(1) # Prevent Index out of range just before the optimal solution is detected. 

    rng = get_default_rng(seed)

    x: OneMax = problem(n, rng=rng)   
    f_x = x.fitness
    lbd = lbds[f_x]
    
    total_evals = 1 # total number of solution evaluations    

    mtimes, ctimes = [], []
    steps = 1
    old_f_x = f_x
    while not x.is_optimal():
        # mutation phase
        s = time.time()  
        p = lbd/n
        xprime, f_xprime, ne1 = x.mutate(p, round(lbd), rng)      
        mtimes.append(time.time()-s)     

        # crossover phase
        s = time.time()
        c = 1/lbd          
        y, f_y, ne2 = x.crossover(xprime, c, round(lbd), include_xprime_crossover, count_different_inds_only,  rng)          
        ctimes.append(time.time()-s)                     

        # selection phase
        old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y
        lbd = lbds[f_x]

        total_evals = total_evals + ne1 + ne2
        steps += 1
            
        if total_evals>=max_evals:
            max_evals *= 2
            break
        
    return x, f_x, total_evals #, mtimes, ctimes        

def onell_dynamic_theory(n, problem=OneMax, seed=None,
                    max_evals = 99999999,
                    count_different_inds_only=True,                    
                    include_xprime_crossover = True           
                ):
    """
    (1+LL)-GA, dynamic version with theoretical results
    lbd = sqrt(n*(n-f(x))), p = lbd/n, c=1/lbd    

    """
    rng = get_default_rng(seed)

    x = problem(n, rng=rng)   
    f_x = x.fitness
    
    total_evals = 1 # total number of solution evaluations    

    mtimes, ctimes = [], []
    steps = 1
    old_f_x = f_x
    while not x.is_optimal():
        # mutation phase
        s = time.time()
        lbd = np.sqrt(n / (n-f_x))        
        p = lbd/n
        xprime, f_xprime, ne1 = x.mutate(p, round(lbd), rng)      
        mtimes.append(time.time()-s)     

        # crossover phase
        s = time.time()
        c = 1/lbd          
        y, f_y, ne2 = x.crossover(xprime, c, round(lbd), include_xprime_crossover, count_different_inds_only,  rng)          
        ctimes.append(time.time()-s)                     

        # selection phase
        old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y

        total_evals = total_evals + ne1 + ne2

        #print("%d: evals=%d; x=%d; xprime=%d; y=%d; obj=%d; p=%.2f; c=%.2f; lbd=%.2f" % (steps, total_evals, old_f_x,xprime.fitness, y.fitness, x.fitness, p, c, lbd))        
        
        #if steps==1:
        #    print("steps,total_evals,old_f_x,p,c,lbd")
        #print("%d,%d,%d,%.2f,%.2f,%.2f" % (steps, total_evals, old_f_x, p, c, lbd))        

        steps += 1
            
        if total_evals>=max_evals:
            total_evals *= 2
            break  

    #print(total_evals)
        
    return x, f_x, total_evals #, mtimes, ctimes        

def onell_lbd_one(n, problem=OneMax, seed=None,
                    max_evals = 99999999,
                    count_different_inds_only=True,                    
                    include_xprime_crossover = True           
                ):
    """
    (1+LL)-GA, with lbd=1 for all iterations
    p = lbd/n, c=1/lbd    

    """
    rng = get_default_rng(seed)

    x = problem(n, rng=rng)   
    f_x = x.fitness
    
    total_evals = 1 # total number of solution evaluations    

    mtimes, ctimes = [], []
    steps = 1
    old_f_x = f_x
    lbd = 1
    while not x.is_optimal():
        # mutation phase
        s = time.time()
        #lbd = np.sqrt(n / (n-f_x))        
        p = lbd/n
        xprime, f_xprime, ne1 = x.mutate(p, round(lbd), rng)      
        mtimes.append(time.time()-s)     

        # crossover phase
        s = time.time()
        c = 1/lbd          
        y, f_y, ne2 = x.crossover(xprime, c, round(lbd), include_xprime_crossover, count_different_inds_only,  rng)          
        ctimes.append(time.time()-s)                     

        # selection phase
        old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y

        total_evals = total_evals + ne1 + ne2

        #print("%d: evals=%d; x=%d; xprime=%d; y=%d; obj=%d; p=%.2f; c=%.2f; lbd=%.2f" % (steps, total_evals, old_f_x,xprime.fitness, y.fitness, x.fitness, p, c, lbd))        
        
        #if steps==1:
        #    print("steps,total_evals,old_f_x,p,c,lbd")
        #print("%d,%d,%d,%.2f,%.2f,%.2f" % (steps, total_evals, old_f_x, p, c, lbd))        

        steps += 1
            
        if total_evals>=max_evals:
            break  

    #print(total_evals)
        
    return x, f_x, total_evals #, mtimes, ctimes 

def rls_optimal_lo(n, problem=LeadingOne, seed=None, max_evals = 99999999):
    """
    RLS with optimal step size (k=int(n/f(x) + 1)) for Leading One problem
    Reference: https://dl.acm.org/doi/10.1145/3205455.3205560 (Lemma 1)
    """
    rng = get_default_rng(seed)

    x = LeadingOne(n, rng=rng)   
    f_x = x.fitness
    
    total_evals = 1 # total number of solution evaluations    
        
    old_f_x = f_x
    total_evals = 0
    while not x.is_optimal():
        k = int(n / (f_x + 1))        
        y, f_y = x.mutate_rls(k, rng)
        total_evals += 1                           

        # selection phase
        old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y       

        #print("%d: evals=%d; obj=%d; k=%d" % (total_evals, total_evals, f_x, k))
            
        if total_evals>=max_evals:
            break  

    #print(total_evals)
        
    return x, f_x, total_evals

def rls(n, problem=LeadingOne, seed=None, max_evals = 99999999):
    """
    RLS with optimal step size (k=int(n/f(x) + 1)) for Leading One problem
    Reference: https://dl.acm.org/doi/10.1145/3205455.3205560 (Lemma 1)
    """
    rng = get_default_rng(seed)

    x = LeadingOne(n, rng=rng)   
    f_x = x.fitness
    print(f_x)
    
    total_evals = 1 # total number of solution evaluations    
        
    old_f_x = f_x
    total_evals = 0
    while not x.is_optimal():
        k = rng.choice([1,2,4],size=1) 
        y, f_y = x.mutate_rls(k, rng)
        total_evals += 1                           

        # selection phase
        old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y       

        #print("%d: evals=%d; obj=%d; k=%d" % (total_evals, total_evals, f_x, k))
            
        if total_evals>=max_evals:
            break  

    #print(total_evals)
        
    return x, f_x, total_evals
