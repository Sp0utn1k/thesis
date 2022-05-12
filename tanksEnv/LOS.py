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

def generate_vis_grid(size):
    visibility=size
    vis_grid = []
    for x in range(0,visibility+1):
        for y in range(0,visibility+1):
            if [x,y] != [0,0] and math.sqrt(x**2+y**2) <= visibility:
                vis_grid.append([x,y])
                if x:
                    vis_grid.append([-x,y])
                    if y:
                        vis_grid.append([-x,-y])
                if y:
                    vis_grid.append([x,-y])
    return vis_grid

def hidden(pos,vis=20):

    hidden = []
    vis_grid = generate_vis_grid(vis)
    for [x,y] in vis_grid:
        LOS = los([0,0],[x,y])
        if pos in LOS:
            hidden.append([x,y])
    return hidden


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


    # with open(filename, 'rb') as file:
    #     LOS = pickle.load(file)
    # vect1,vect2 = [20,5],[7,12]
    # los1 = los(vect1,vect2)
    # los1.sort() 
    # print(los1)
    # los2 = get_los(vect1,vect2,LOS)
    # los2.sort()
    # print(los2)


    visibility=20
    filename = f'hidden_cells{visibility}.pkl'    
    # hidden_dict = {}
    # for i,pos in enumerate(generate_vis_grid(visibility)):
    #     print(i)
    #     x,y = pos
    #     if x >= 0 and y >= 0:
    #         hidden_pos = hidden(pos,vis=visibility)
    #         hidden_dict[x,y] = hidden_pos
    #         if x:
    #             hidden_dict[-x,y] = [[-x,y] for [x,y] in hidden_pos]
    #             if y:
    #                 hidden_dict[-x,-y] = [[-x,-y] for [x,y] in hidden_pos]
    #         if y:
    #             hidden_dict[x,-y] = [[x,-y] for [x,y] in hidden_pos]

    # with open(filename, 'wb') as file:
    #     pickle.dump(hidden_dict, file)

    with open(filename, 'rb') as file:
        hidden_dict = pickle.load(file)

    grid = generate_vis_grid(5)
    obstacles = [[0,1],[0,-1],[1,0],[-1,0]]
    hiddens = []
    for obs in obstacles:
        hiddens += hidden_dict[tuple(obs)]

    print(grid,'\n')
    grid = [pos for pos in grid if pos not in hiddens and pos not in obstacles]
    print(grid)