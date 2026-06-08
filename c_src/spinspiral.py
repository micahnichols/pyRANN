import sympy as sp
from sympy import sin,cos,Symbol,Function
from numpy import array

def rotationmatrix(axis,angle):
    return array([[cos(angle)+axis[0]**2*(1-cos(angle)), axis[0]*axis[1]*(1-cos(angle))-axis[2]*sin(angle), axis[0]*axis[2]*(1-cos(angle))+axis[1]*sin(angle)],
        [axis[1]*axis[0]*(1-cos(angle))+axis[2]*sin(angle), cos(angle)+axis[1]**2*(1-cos(angle)), axis[1]*axis[2]*(1-cos(angle))-axis[0]*sin(angle)],
        [axis[2]*axis[0]*(1-cos(angle))-axis[1]*sin(angle), axis[2]*axis[1]*(1-cos(angle))+axis[0]*sin(angle), cos(angle)+axis[2]**2*(1-cos(angle))]])

ax = Symbol('ax')
ay = Symbol('ay')
az = Symbol('az')
sxi = Symbol('sxi')
syi = Symbol('syi')
szi = Symbol('szi')
angle = Symbol('theta')
axis = array([ax,ay,az])
si = array([sxi,syi,szi])
s = rotationmatrix(axis,angle)@si
