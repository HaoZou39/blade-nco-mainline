import numpy as np

from src.qd.acquisition import centroid_global_acquire
from src.qd.archive import CurrentSurrogateScore


def test_centroid_floor_then_global_gp_fill():
    centers=np.zeros((1000,4)); centers[1]=1
    hashes=["a","b","c","d","e"]
    niches={"a":0,"b":0,"c":1,"d":1,"e":1}
    qd={"a":(.2,.2,.2,.2),"b":(.1,.1,.1,.1),"c":tuple([.9]*4),"d":tuple([.8]*4),"e":tuple([.7]*4)}
    scores={h:CurrentSurrogateScore(v,"M") for h,v in zip(hashes,[100,1,2,50,40])}
    result=centroid_global_acquire(hashes,scores,niches,qd,centers,4,current_score_version="M")
    assert [x.instance_hash for x in result[:2]]==["b","c"]
    assert [x.instance_hash for x in result[2:]]==["a","d"]
    assert all(x.component=="centroid_floor" for x in result[:2])
