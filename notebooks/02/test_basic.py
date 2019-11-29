from basicmath import div

def test_div():
    assert div(4, 2) == 2

def test_div_2():
    assert div(15, 3) == 5


def test_div_for_float():
    assert div(1, 2) == 0.5









from basicmath import sortlist

def test_srt():
    l = [15, 2, 3, 7, 6, 0]
    l_sorted = [0, 2, 3, 6, 7, 15]

    assert sortlist(l) == l_sorted










from basicmath import openfile

def test_open():
    filename = "missing-file.txt" 
    try: 
        openfile(filename) 
        raise AssertionError("Expected exception not thrown.")

    except FileNotFoundError: 
        pass 

    
