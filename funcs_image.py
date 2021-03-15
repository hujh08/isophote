#!/usr/bin/env python3

'''
    functions of image

    including
        functions of interpolation
        intensity derivative in image
'''

import numbers

from funcs_linfit import poly_2d

# interpolation functions
def sample_ellipse(img, xy0, a, ba, pa, num=100):
    '''
        sample along an ellipse
    '''
    pass

## coordinates for a ellipse
def coords_of_ellipse_angles(angles, xy0, a, ba, pa):
    '''
        coordinates (x, y) for angles
            along an ellipse defined by (xy0, a, ba, pa)

        also used in function `funcs_modify:modify_ellipse`
    '''
    # major-minor axes (x-axis along major)
    assert 0<ba<=1
    b=ba*a

    xs_mm=a*np.cos(angles)
    ys_mm=b*np.sin(angles)

    # convert to real axis
    x0, y0=xy0

    cosp=np.cos(pa)
    sinp=np.sin(pa)

    xs=x0+xs_mm*cosp-ys_mm*sinp
    ys=y0+xs_mm*sinp+ys_mm*cosp

    return xs, ys

# drivative of intensity
def deriv_around_point(img, xy, semiw=2, order=2):
    '''
        calculate derivative vector (dI/dx, dI/dy) of intensify
            around a given point (x, y)

        `semiw`: semi-width to inspect for the calculation

        fit to data in a region with polynomial function (e.g. order=2):
            I = I0 + A1 x + B1 y + A2 x^2 + B2 y^2 + C2 xy

        then at (x, y)
            dI/dx = A1 + 2 A2 x + C2 y
            dI/dy = B1 + 2 B2 y + C2 x

        More simply, shift the coordinate to be centered at (x, y) before fit
        then, at new coordinates, (x, y) becomes (0, 0)
            dI/dx = A1
            dI/dy = B1

        Furthermore, higher order polynomial is supported
    '''
    x, y=xy

    # convert to integral
    xi=int(np.round(x))
    yi=int(np.round(y))

    assert semiw>=1  # at least 9 pixel
    semiw=int(np.ceil(semiw))

    # extract pixel for calculation
    data=img[(yi-semiw):(yi+semiw+1), (xi-semiw):(xi+semiw+1)]
    assert data.size>9 # expect at least 9 pixel

    ## number of coefficient
    assert order>=2  # at least 2 order
    ncoef=(order+2)*(order+1)/2
    assert data.size>ncoef

    # coordinates
    ny, nx=data.shape
    xs=np.arange(nx)+xi-semiw-x
    ys=np.arange(ny)+yi-semiw-y

    # polynomial fit: I = A0 + A1 x + A2 y + ...
    A0, A1, A2, *_=poly_2d(data, (xs, ys), order=order)

    # return dI/dx, dI/dy
    return A1, A2

def deriv_along_direction(img, xy, pa, **kwargs):
    '''
        calculate derivation around a point xy,
            along a direction (defined by pa)
    '''
    # direction
    if isinstance(pa, numbers.Number)
        dx=np.cos(pa)
        dy=np.sin(pa)
    else:
        dx, dy=pa

        d=np.hypot(dx, dy)
        assert d>0

        dx, dy=dx/d, dy/d

    # derivation vector
    dIdx, dIdy=deriv_around_point(img, xy, **kwargs)

    return dIdx*dx+dIdy*dy

def deriv_perpend_direction(img, xy, pa, **kwargs):
    '''
        calculate derivation around a point xy,
            perpendicular to a direction (defined by pa)

        if the direction is (dx, dy), then perpendicular direction is (-dy, dx)
    '''
    # direction
    if isinstance(pa, numbers.Number)
        dx=np.cos(pa)
        dy=np.sin(pa)
    else:
        dx, dy=pa

        d=np.hypot(dx, dy)
        assert d>0

        dx, dy=dx/d, dy/d

    # derivation vector
    dIdx, dIdy=deriv_around_point(img, xy, **kwargs)

    # perpendicular direction is (-dy, dx)
    return -dIdx*dy+dIdy*dx
