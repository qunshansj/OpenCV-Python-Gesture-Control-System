python

def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0] #关键点1 x1
    v1_y=v1[1] #关键点1 y1
    v2_x=v2[0] #关键点2 x2
    v2_y=v2[1] #关键点2 y2
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5)))) #计算手指角度 将弧度转化为度数
    except:
        angle_ = 65535.
    if angle_ > 180.: #如果角度大于180度 则
        angle_ = 65535.
    return angle_
def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = [] #保存各个手指的弯曲角度
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle( #输入大拇指多组 两个关键点 并计算各组关键点夹角的度数
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1

