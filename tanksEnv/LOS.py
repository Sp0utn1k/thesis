import pickle, math
  

def norm(vect1,vect2):
    x1,y1 = vect1
    x2,y2 = vect2
    res = (x2-x1)**2 + (y2-y1)**2
    return math.sqrt(res)


def los(vect1,vect2):
    [x1,y1] = vect1
    [x2,y2] = vect2
    N = 100*round(norm(vect1,vect2))
    dy = (y2-y1)/N
    dx = (x2-x1)/N

    x_iter = [round(x1+i*dx) for i in range(N)]
    y_iter = [round(y1+i*dy) for i in range(N)]

    LOS = list(set([(xi,yi) for (xi,yi) in zip(x_iter,y_iter)]))

    LOS = [[x,y] for (x,y) in LOS if [x,y] != vect1 and [x,y] != vect2]
    return LOS

def get_los(vect1,vect2,los_dict):

    if vect2[0] < vect1[0]:
        vect1,vect2 = vect2,vect1

    diff = [vect2[0]-vect1[0],vect2[1]-vect1[1]]
    mirrored = False
    if diff[1] < 0:
        mirrored = True
        diff[1] = -diff[1]

    los = [[i+vect1[0],j*(-1)**mirrored+vect1[1]] for [i,j] in los_dict[tuple(diff)]]
    return los


if __name__ == "__main__":
    sz = 100
    filename = f'los{sz}.pkl'

    # LOS = {}
    # for i in range(sz):
    #     for j in range(sz):
    #         if (i,j) == (0,0):
    #             continue
    #         LOS[i,j] = los([0,0],[i,j])

    # with open(filename, 'wb') as file:
    #     pickle.dump(LOS, file)


    with open(filename, 'rb') as file:
        LOS = pickle.load(file)

    vect1,vect2 = [20,5],[7,12]

    los1 = los(vect1,vect2)
    los1.sort() 
    print(los1)

    los2 = get_los(vect1,vect2,LOS)
    los2.sort()
    print(los2)