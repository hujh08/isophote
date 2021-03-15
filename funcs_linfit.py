#!/usr/bin/env python3

'''
    functions for linear regression

    including:
        `lin_regress`: a general multiple linear regression
        `poly_multiple`: multiple polynomial fit to an order
        `poly_2d`: polynomial fit to 2d data
        `fit_trigon_angle`: fit with a sum of trigonometric functions to an order
'''

import numpy as np

# multiple linear regression
def lin_regress(ys, *xs, fit_intercept=True):
    '''
        linear regression:
            y=I0 + A1 x1 + A2 x2 + ...

        return co-efficients, I0, A1, A2, ...

        if not `fit_intercept`, only fit with y=A1 x1 + A2 x2 + ...
            and only A1, A2, ... are returned
    '''
    ys=np.asarray(ys)

    assert len(xs)>=1  # at least 1 x-array
    xs=np.column_stack([*xs])
    if fit_intercept:
        xs=np.column_stack([np.ones(len(ys)), xs])

    return np.linalg.lstsq(xs, ys, rcond=None)[0]

# multiple polynomial fit
def iter_dict_order(m, s):
    '''
        iter a m-elements tuple (i1, i2, ..., im)
            with sum of elements is s, that is i1+i2+...+im=s

        order is inverse of dictionary order
            that means
                when (i1, i2,..., ik)=(j1, j2, ..., jk) and i(k+1)>j(k+1)
                    then yield (i1, ..., im) before (j1, j2, ..., jm)
    '''
    assert s>=0 and m>=1

    if m==1:
        yield (s,)
    elif s==0:
        yield (0,)*m
    else:
        for i in range(s, -1, -1):
            for others in iter_dict_order(m-1, s-i):
                yield (i, *others)

def poly_multiple(ys, *xs, order=2):
    '''
        mupliple polynomial fit

        Let x1, x2, ... for data in xs

        Terms with same order are sorted as decreasing power index of x1, then x2, ...
            for example, x1^3, x1^2 x2, x1 x2^2, x2^3
        This is also refered to inverse of dictionary order

        y = A0 + A1 x1 + A2 x2 + A3 x1^2 + A4 x1 x2 + A5 x2^2 + ...

        return A0, A1, A2, ...
    '''
    assert len(xs)>=1   # at least one x
    n=len(ys)
    m=len(xs)

    # x1^k, y2^k, ..., k=0, 1, ..., order
    xks=[]
    for xis in xs:
        assert len(xis)==n  # same length with ys
        xks.append([np.ones(n)])

        xkis=xks[-1]
        for i in range(1, order+1):
            xkis.append(xkis[-1]*xis)

    xterms=[]
    for o in range(1, order+1):
        for ks in iter_dict_order(m, o):
            t=xks[0][ks[0]]
            for xk, k in zip(xks[1:], ks[1:]):
                t=t*xk[k]
            xterms.append(t)

    coefs=lin_regress(ys, *xterms)

    return coefs

# polynomial fit to 2d data
def poly_2d(data, xys, coording='xy', order=2):
    '''
        polynomial fit to 2d data

        xs, ys: 1D or 2D arrays for the coordinates of data

        Terms with same order are sorted as decreasing power index of x
            for example, x^3, x^2 y, x y^2, y^3

        I = A0 + A1 x + A2 y + A3 x^2 + A4 xy + A5 y^2 + ...

        return A0, A1, A2, ...

        Parameter:
            coording: 'xy', 'center', 'region'
                how to use xys
                'xy': coordinates array given in `xys`
                'center': (x0, y0, dx, dy) for `xys`
                    (x0, y0) is the pixel for coordination (0, 0)

                    xs=(i-x0)*dx, ys=(j-y0)*dy
                        for pixel (j, i)
                'corner': (x1, y1, dx, dy) for `xys`
                    (x1, y1) is the coordination of left-low pixel

                    xs=x1+i*dx, ys=y1+j*dy

                `region`: (x1, x2, y1, y2) for `xys`
    '''
    data=np.asarray(data)
    shape=data.shape
    nd=len(shape)   # number of dimensions
    n=data.size

    if coording=='xy':
        xs, ys=xys
    else:
        assert nd>1   # only work for well-aligned data
        ny, nx=shape

        if coording=='center':
            x0, y0, dx, dy=xys
            x1, y1=-x0*dx, -y0*dy

        elif coording=='region':
            x1, x2, y1, y2=xys

            if nx==1:
                assert x1==x2
                dx=1
            else:
                dx=(x2-x1)/(nx-1)

            if ny==1:
                assert y1==y2
                dy=1
            else:
                dy=(y2-y1)/(ny-1)

        elif coording=='corner':
            x1, y1, dx, dy=xys

        else:
            raise Exception('unexpected xys:', xys)

        xs=np.arange(nx)*dx+x1
        ys=np.arange(ny)*dy+y1

    data=data.ravel()
    xs=np.asarray(xs).ravel()
    ys=np.asarray(ys).ravel()

    if nd>1:  # high dimension
        ny, nx=shape

        if len(xs)!=n:
            assert len(xs)==nx
            xs=np.ravel(np.ones((ny, 1))*np.reshape(xs, (1, nx)))

        if len(ys)!=n:
            assert len(ys)==ny
            ys=np.ravel(np.reshape(ys, (ny, 1))*np.ones((1, nx)))

    return poly_multiple(data, xs, ys, order=order)

# fit with trigonometric series of azimuthal angle to an order
def fit_trigon_angle(vals, angles, order=2):
    '''
        fit with trigonometric series of azimuthal angle to an order
            I=I0 + A1 sin(a) + B1 cos(a) + A2 sin(2a) + B2 cos(2a) + ...

        return I0, A1, B1, ...
    '''
    sincoss=[]
    for i in range(1, order+1):
        sincoss.append(np.sin(i*angles))
        sincoss.append(np.cos(i*angles))

    return lin_regress(vals, *sincoss)
