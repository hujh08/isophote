#!/usr/bin/env python3

'''
    functions to modify ellipse according to the coefficients of fitting result

    ellipse with a fixed semi-major axis length, `a` is given by parameters
        (x0, y0, ba, pa)
        where
            (x0, y0): center
            ba: b/a
            pa: position angle of major axis

    fitting is done to a sample along the ellipse:
        I=I0 + A1 sin(a) + B1 cos(a) + A2 sin(2a) + B2 cos(2a)
    Modification here only uses coefficients (A1, B1, A2, B2)

    all functions return (dx, dy, dba, dpa)
        which are midification to (x0, y0, ba, pa)
'''

from funcs_image import deriv_along_direction, deriv_perpend_direction, coords_of_ellipse_angles

# for sample along circle: ba=1
def modify_x_y_ba(coefs, img, a, xy0, ba, pa,
                    semiw=2, correct_guarded=True, **kwargs):
    '''
        modification of (x, y, ba)

        extract these modifications (no pa modifying included) to a function,
            since they are used in two functions, `modify_circle`, `modify_ellipse`

        only use (A1, B1, B2) in this function

        Parameter
            semiw: semi-width to inspect for derivation calculation
                it is also used to restrict the correction not too large

            correct_guarded: bool,
                if True, guarded correction
                    that means correction of xy cannot be larger than `semiw`
    '''
    A1, B1, B2=coefs
    # ellipse of previous sample
    x0, y0=xy0

    cosp=np.cos(pa)
    sinp=np.sin(pa)

    b=ba*a

    kwargs['semiw']=semiw

    ## along major axis
    x=x0+a*cosp  # one endpoint of major axis
    y=y0+a*sinp

    dI1=deriv_along_direction(img, (x, y), pa=pa, **kwargs)
    dI2=deriv_along_direction(img, (2*x0-x, 2*y0-y), pa=pa+np.pi, **kwargs)
    dmajor=-2*B1/(dI1+dI2)

    if correct_guarded and np.abs(dmajor)>semiw:
        dmajor=np.sign(dmajor)*semiw

    ## along minor axis
    x=x0-b*sinp
    y=y0+b*cosp

    dI3=deriv_along_direction(img, (x, y), pa=pa+np.pi/2, **kwargs)
    dI4=deriv_along_direction(img, (2*x0-x, 2*y0-y), pa=pa-np.pi/2, **kwargs)
    dminor=-2*A1/(dI3+dI4)

    if correct_guarded and np.abs(dminor)>semiw:
        dminor=np.sign(dminor)*semiw

    ## convert to x-y axes
    dx=dmajor*cosp-dminor*sinp
    dy=dmajor*sinp+dminor*cosp

    ## to ba
    dba=2*B2/(r*(dI3+dI4)/2)

    if correct_guarded:
        if dba>0:
            dba=0
        elif np.abs(dba*r)>semiw:
            dba=-semiw/r

    return dx, dy, dba

def modify_circle(coefs, img, r, xy0, pa0=0, **kwargs):
    '''
        modification accroding to result of fitting to a sample along a circle

        coefs is obtained from fitting to a sample along a circle
            I=I0 + A1 sin(a) + B1 cos(a) + A2 sin(2a) + B2 cos(2a)

        Firstly, rotate to PA
            I=I0 + A1' sin(a-p) + B1' cos(a-p) + B2' cos(2(a-p))
        which means
            B2=B2' cos(2p),        A2=B2' sin(2p)
            A1= A1' cos(p) + B1' sin(p)
            B1=-A1' sin(p) + B1' cos(p)

        So
            A1'= A1 cos(p) - B1 sin(p)
            B1'= A1 sin(p) + B1 cos(p)

        Parameter:
            r, xy0, pa0: define a circle used in previous sample
                `pa0` is optional
                    which is the starting angles for the sample angles

        Optional keyword arguments `kwargs` are for derivative calculation
    '''
    A1, B1, A2, B2=coefs

    # rotate to align with position angle
    B2n=np.hypot(A2, B2)

    pa2=np.arccos(B2/B2n)
    if A2<0:
        pa2=2*np.pi-pa2
    pa=pa2/2

    # new coefficients
    cosp=np.cos(pa)
    sinp=np.sin(pa)
    
    A1n=A1*cosp-B1*sinp
    B1n=A1*sinp+B1*cosp

    # modification
    ## to pa
    dpa=pa
    pa=pa0+dpa

    ## to (x, y), ba
    dx, dy, dba=modify_x_y_ba((A1n, B1n, B2n), img, r, xy0, 1, pa, **kwargs)

    return dx, dy, dba, dpa

def modify_ellipse(coefs, img, a, xy0, ba, pa,
                    semiw=2, correct_guarded=True, **kwargs):
    '''
        modification accroding to result of fitting to a sample along an ellipse
        
        coefs is obtained from fitting to a sample along a ellipse
            I=I0 + A1 sin(a) + B1 cos(a) + A2 sin(2a) + B2 cos(2a)

        Parameter:
            a, xy0, ba, pa: define a ellipse used in previous sample
                `pa0` is optional
                    which is the starting angles for the sample angles

        Optional keyword arguments `kwargs` are for derivative calculation
    '''
    kwargs['semiw']=semiw
    kwargs['correct_guarded']=correct_guarded

    # round to a circle
    assert 0<ba<=1
    if a*(1-ba)<1:    # b-a less than 1 pixel; treating as a circle
        return modify_circle(coefs, img, a, xy0, pa0=pa, **kwargs)

    # for ellipse
    A1, B1, A2, B2=coefs

    x0, y0=xy0

    cosp=np.cos(pa)
    sinp=np.sin(pa)

    ## dx, dy, dba
    dx, dy, dba=modify_x_y_ba((A1, B1, B2), img, a, xy0, ba, pa, **kwargs)

    ## dpa
    del kwargs['correct_guarded']

    ### derivative at points with t=pi/4+k*pi/2; x=a*cos(t), y=b*sin(t)
    x, y=coords_of_ellipse_angles(np.pi/4, (0, 0), a, ba, 0)
    r=np.hypot(x, y)

    xcosp, xsinp=x*cosp, x*sinp
    ycosp, ysinp=y*cosp, y*sinp

    x1, y1= xcosp-ysinp,  xsinp+ycosp
    x2, y2=-xcosp-ysinp, -xsinp+ycosp
    x3, y3=-xcosp+ysinp, -xsinp-ycosp
    x4, y4= xcosp+ysinp,  xsinp-ycosp

    dI1=deriv_perpend_direction(img, (x0+x1, y0+y1), pa=(x1, y1), **kwargs)
    dI2=deriv_perpend_direction(img, (x0+x2, y0+y2), pa=(x2, y2), **kwargs)
    dI3=deriv_perpend_direction(img, (x0+x3, y0+y3), pa=(x3, y3), **kwargs)
    dI4=deriv_perpend_direction(img, (x0+x4, y0+y4), pa=(x4, y4), **kwargs)

    dI=((dI1-dI2)+(dI3-dI4))/2
    dpa=-2*A2/(r*dI)

    if correct_guarded and np.abs(r*dpa)>semiw:
        dpa=np.sign(dpa)*semiw/r

    return dx, dy, dba, dpa