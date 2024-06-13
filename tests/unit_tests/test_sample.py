import src.sample as smp

# These tests test the methods from sample.py
def test_sign_pos():
    assert smp.sign(89) == 1 , "Should be 1"

def test_sign_neg():
    assert smp.sign(-971) == -1 , "Should be -1"

def test_treatment_pos():
    assert smp.treatment(89) == 1 , "Should be 1"

def test_treatment_neg():
    assert smp.treatment(-971) == 0 , "Should be 0"
    
def test_indicator_pos():
    assert smp.indicator(0.089) == 1 , "Should be 1"

def test_indicator_neg():
    assert smp.indicator(-971) == 0 , "Should be 0"

def test_genT_size():
    X = [0,0,0,1,1,1]
    T = smp.genT(X)

    assert len(T) == 6 , 'Lenght should be 6'

def test_genT_value():
    X = [-0.5,-1.64,-0.3,2,1.2,1.5]
    T = smp.genT(X)
    print(T)
    assert all(T == [0,0,0,1,1,1]) , 'Treatment is not correctly assigned'

def test_mu_noack_1():
    assert smp.mu_noack(0,0) == 0 ,'Should be zero'

def test_mu_noack_2():
    assert smp.mu_noack(0,2) == 4 ,'Should be four'

def test_mu_noack_3():
    assert smp.mu_noack(0,0.01) == 0.0001 , 'Should be 0.0001'

def test_mu_noack_4():
    assert smp.mu_noack(40,0.01) == 0.07610000000000003 , 'Should be 0.07610000000000003'

def test_genY_noack_value():
    X = [-2,-1,0,0.5,1,2]
    E = [0,0,0,0,0,0]
    Y = smp.genY_noack(40,X,E)
    assert all(Y==[-4,-1,0,0.25,1,4])
