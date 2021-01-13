''''
# [18/11/2020] -- Modifications by Adam Gaier (adam.gaier@autodesk.com)
#
#          Copyright Marc Bodmer 2013.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)


Python library for parametric representation of airfoils according to PARSEC (PARametric SECtion).

References:

- Parametric Airfoils and Wings, by Helmut Sobieczky,
  http://www.sobieczky.at/aero/literature/H141.pdf

- Geometric Parameterisation and Aerodynamic Shape Optimisation, by Feng Zhu
  http://etheses.whiterose.ac.uk/6704/1/Feng_Zhu_Thesis_final.pdf

- An airfoil shape optimization technique coupling PARSEC parameterization and evolutionary algorithm, Veccia, Daniele, D'Amato
  https://www.iris.unina.it/retrieve/handle/11588/581560/9503/1-s2.0-S1270963813002046-full_article.pdf
'''

import numpy as np
import math

class Parameters(object):
    '''
    12 Parameters defining a PARSEC airfoil are defined as values between 0 and 1
      corresponsing to the min/max of a range.
    
    NOTE: We are only using 10, and leaving the coordinate and thickness of the trailing edge alone

    Example Params:
      RAE  2822: 
      NACA 0012: 
      NACA 2412: 

    '''

    def __init__(self,genome=None):
      param_vector = np.array((0.1155, 0.7695, 0.1391, 0.2788, 0.1244, 0.1516, 0.7519, 0.3076, 0.5116, 0, 0.2629, 0.7630)) #RAE2822 from MATLAB
      #param_vector = np.array((0.1155, 0.7695, 0.180, 0.2788, 0.1244, 0.1516, 0.720, 0.5576, 0.5116, 0.5, 0.25, 0.7)) #RAE2822 (for Python my closest fit by hand)
      if genome is not None:
        param_vector[[0,1,2,3,4,5,6,7,10,11]] = genome
       
      # Parameter ranges (found by trial and error)
      r_min = np.array((0.0037, 0.1500, 0.0440,-1.0000, 0.0037, 0.3000,-0.1200,-1.5000,-0.0100, 0.0020,-10.0000,-10.0000))
      r_max = np.array((0.0500, 0.5175, 0.1588, 0.3000, 0.0500, 0.6000,-0.0400,-0.3000, 0.0100, 0.0026, 10.0000, 20.0000))
      spread = r_max-r_min
      p = (param_vector*spread)+r_min

      self.r_le_up    =  p[0]                 # Upper leading edge radius
      self.X_up       =  p[1]                 # Upper crest location X coordinate
      self.Z_up       =  p[2]                 # Upper crest location Z coordinate
      self.Z_XX_up    =  p[3]                 # Upper crest location curvature
      self.r_le_lo    =  p[4]                 # Lower leading edge radius
      self.X_lo       =  p[5]                 # Lower crest location X coordinate
      self.Z_lo       =  p[6]                 # Lower crest location Z coordinate
      self.Z_XX_lo    = -p[7]                 # Lower crest location curvature
      self.Z_te       =  p[8]                 # Trailing edge Z coordinate
      self.dZ_te      =  p[9]                 # Trailing edge thickness
      self.alpha_te   =  math.radians(p[10])  # Trailing edge direction angle
      self.beta_te    =  math.radians(p[11])  # Trailing edge wedge angle
    
    def __str__(self):
        rep = f'''
        PARSEC-11 airfoil parameters:
        -------------------------------------------------------------
        Upper Leading edge radius [r_le]:         {self.r_le_up}
        Upper crest location X coordinate [X_up]: {self.X_up}
        Upper crest location Z coordinate [Z_up]: {self.Z_up}
        Upper crest location curvature [Z_XX_up]: {self.Z_XX_up}
        Lower Leading edge radius [r_le]:         {self.r_le_lo}        
        Lower crest location X coordinate [X_lo]: {self.X_lo}
        Lower crest location Z coordinate [Z_lo]: {self.Z_lo}
        Lower crest location curvature [Z_XX_lo]: {self.Z_XX_lo}
        Trailing edge Z coordinate [Z_te]:        {self.Z_te}
        Trailing edge thickness [dZ_te]:          {self.dZ_te}
        Trailing edge direction angle [alpha_te]: {self.alpha_te}
        Trailing edge wedge angle [beta_te]:      {self.beta_te}
        -------------------------------------------------------------
        '''
        return rep


class Coefficients(object):
    '''
    This class calculates the equation systems which define the coefficients
    for the polynomials given by the parsec airfoil parameters.
    '''
    def __init__(self, parsec_params):
        self._a_up = self._calc_a_up(parsec_params)
        self._a_lo = self._calc_a_lo(parsec_params)
    
    def a_up(self):
        '''Returns coefficient vector for upper surface'''
        return self._a_up
    
    def a_lo(self):
        '''Returns coefficient vector for lower surface'''
        return self._a_lo
    
    def _calc_a_up(self, parsec_params):
        Amat = self._prepare_linsys_Amat(parsec_params.X_up)
        Bvec = np.array([parsec_params.Z_te + parsec_params.dZ_te/2, parsec_params.Z_up,
                            math.tan(parsec_params.alpha_te - parsec_params.beta_te/2),
                            0.0, parsec_params.Z_XX_up, math.sqrt(2*parsec_params.r_le_up)]) 
        return np.linalg.solve(Amat, Bvec)
    
    def _calc_a_lo(self, parsec_params):
        Amat = self._prepare_linsys_Amat(parsec_params.X_lo)
        # Bvec = np.array([parsec_params.Z_te - parsec_params.dZ_te/2, parsec_params.Z_lo,
        #                     math.tan(parsec_params.alpha_te + parsec_params.beta_te/2),
        #                     0.0, parsec_params.Z_XX_lo, -math.sqrt(2*parsec_params.r_le_lo)])
        Bvec = np.array([-parsec_params.Z_te + parsec_params.dZ_te/2, -parsec_params.Z_lo,
                            math.tan(parsec_params.alpha_te + parsec_params.beta_te/2),
                            0.0, -parsec_params.Z_XX_lo, math.sqrt(2*parsec_params.r_le_lo)])                            
        return -np.linalg.solve(Amat, Bvec)
    
    def _prepare_linsys_Amat(self, X):
        return np.array(
            [[1.0,           1.0,          1.0,         1.0,          1.0,          1.0        ],
             [X**0.5,        X**1.5,       X**2.5,      X**3.5,       X**4.5,       X**5.5     ],
             [0.5,           1.5,          2.5,         3.5,          4.5,          5.5        ],
             [0.5*X**-0.5,   1.5*X**0.5,   2.5*X**1.5,  3.5*X**2.5,   4.5*X**3.5,   5.5*X**4.5 ],
             [-0.25*X**-1.5, 0.75*X**-0.5, 3.75*X**0.5, 8.75*X**1.5, 13.25*X**2.5, 24.75*X**3.5],
             [1.0,           0.0,          0.0,         0.0,          0.0,          0.0        ]])
            



class Airfoil(object):
    '''Airfoil defined by PARSEC Parameters'''
    def __init__(self, parsec_params=None):
        self.params = Parameters(parsec_params)
        #self._coeff = Coefficients(parsec_params)
        
    def Z_up(self, X):
        '''Returns Z(X) on upper surface, calculates PARSEC polynomial'''
        a = self._coeff.a_up()
        # print(a)
        return a[0]*X**0.5 + a[1]*X**1.5 + a[2]*X**2.5 + a[3]*X**3.5 + a[4]*X**4.5 + a[5]*X**5.5
        
    
    def Z_lo(self, X):
        '''Returns Z(X) on lower surface, calculates PARSEC polynomial'''
        a = self._coeff.a_lo()
        # print(a)
        return a[0]*X**0.5 + a[1]*X**1.5 + a[2]*X**2.5 + a[3]*X**3.5 + a[4]*X**4.5 + a[5]*X**5.5

    def express(self, params=None, n_pts=100):
      '''Returns the [2 X n_pts] matrix of y coords of airfoil'''
      # HACK: just use x-coords of existing airfoil instead of cosspaceing
      if self.params is not None:
        self.params = Parameters(params)
      self._coeff = Coefficients(self.params)
      foil = naca0012()
      foil[1,:n_pts] = self.Z_up(foil[0,:n_pts])
      foil[1,n_pts:] = self.Z_lo(foil[0,n_pts:])
      return foil


  #   0.0091
  #   0.4328
  #   0.0600
  #  -0.6376
  #   0.0095
  #   0.3455
  #  -0.0598
  #  -1.1309
  #   0.0002
  #   0.0020
  #  -4.7411
  #  12.8895

def naca0012():
  return \
  np.array([[9.9975e-01,  9.9901e-01,  9.9778e-01,  9.9606e-01,  9.9384e-01,
             9.9114e-01,  9.8796e-01,  9.8429e-01,  9.8015e-01,  9.7553e-01,
             9.7044e-01,  9.6489e-01,  9.5888e-01,  9.5241e-01,  9.4550e-01,
             9.3815e-01,  9.3037e-01,  9.2216e-01,  9.1354e-01,  9.0451e-01,
             8.9508e-01,  8.8526e-01,  8.7506e-01,  8.6448e-01,  8.5355e-01,
             8.4227e-01,  8.3066e-01,  8.1871e-01,  8.0645e-01,  7.9389e-01,
             7.8104e-01,  7.6791e-01,  7.5452e-01,  7.4088e-01,  7.2699e-01,
             7.1289e-01,  6.9857e-01,  6.8406e-01,  6.6937e-01,  6.5451e-01,
             6.3950e-01,  6.2435e-01,  6.0907e-01,  5.9369e-01,  5.7822e-01,
             5.6267e-01,  5.4705e-01,  5.3139e-01,  5.1570e-01,  5.0000e-01,
             4.8429e-01,  4.6860e-01,  4.5295e-01,  4.3733e-01,  4.2178e-01,
             4.0631e-01,  3.9093e-01,  3.7566e-01,  3.6050e-01,  3.4549e-01,
             3.3063e-01,  3.1594e-01,  3.0143e-01,  2.8711e-01,  2.7300e-01,
             2.5912e-01,  2.4548e-01,  2.3209e-01,  2.1896e-01,  2.0611e-01,
             1.9355e-01,  1.8129e-01,  1.6934e-01,  1.5773e-01,  1.4645e-01,
             1.3552e-01,  1.2494e-01,  1.1474e-01,  1.0492e-01,  9.5492e-02,
             8.6460e-02,  7.7836e-02,  6.9629e-02,  6.1847e-02,  5.4497e-02,
             4.7586e-02,  4.1123e-02,  3.5112e-02,  2.9560e-02,  2.4472e-02,
             1.9853e-02,  1.5708e-02,  1.2042e-02,  8.8560e-03,  6.1560e-03,
             3.9430e-03,  2.2190e-03,  9.8700e-04,  2.4700e-04,  0.0000e+00,
             2.4700e-04,  9.8700e-04,  2.2190e-03,  3.9430e-03,  6.1560e-03,
             8.8560e-03,  1.2042e-02,  1.5708e-02,  1.9853e-02,  2.4472e-02,
             2.9560e-02,  3.5112e-02,  4.1123e-02,  4.7586e-02,  5.4497e-02,
             6.1847e-02,  6.9629e-02,  7.7836e-02,  8.6460e-02,  9.5492e-02,
             1.0492e-01,  1.1474e-01,  1.2494e-01,  1.3552e-01,  1.4645e-01,
             1.5773e-01,  1.6934e-01,  1.8129e-01,  1.9355e-01,  2.0611e-01,
             2.1896e-01,  2.3209e-01,  2.4548e-01,  2.5912e-01,  2.7300e-01,
             2.8711e-01,  3.0143e-01,  3.1594e-01,  3.3063e-01,  3.4549e-01,
             3.6050e-01,  3.7566e-01,  3.9093e-01,  4.0631e-01,  4.2178e-01,
             4.3733e-01,  4.5295e-01,  4.6860e-01,  4.8429e-01,  5.0000e-01,
             5.1570e-01,  5.3139e-01,  5.4705e-01,  5.6267e-01,  5.7822e-01,
             5.9369e-01,  6.0907e-01,  6.2435e-01,  6.3950e-01,  6.5451e-01,
             6.6937e-01,  6.8406e-01,  6.9857e-01,  7.1289e-01,  7.2699e-01,
             7.4088e-01,  7.5452e-01,  7.6791e-01,  7.8104e-01,  7.9389e-01,
             8.0645e-01,  8.1871e-01,  8.3066e-01,  8.4227e-01,  8.5355e-01,
             8.6448e-01,  8.7506e-01,  8.8526e-01,  8.9508e-01,  9.0451e-01,
             9.1354e-01,  9.2216e-01,  9.3037e-01,  9.3815e-01,  9.4550e-01,
             9.5241e-01,  9.5888e-01,  9.6489e-01,  9.7044e-01,  9.7553e-01,
             9.8015e-01,  9.8429e-01,  9.8796e-01,  9.9114e-01,  9.9384e-01,
             9.9606e-01,  9.9778e-01,  9.9901e-01,  9.9975e-01,  1.0000e+00],
           [ 3.6000e-05,  1.4300e-04,  3.2200e-04,  5.7200e-04,  8.9100e-04,
             1.2800e-03,  1.7370e-03,  2.2600e-03,  2.8490e-03,  3.5010e-03,
             4.2160e-03,  4.9900e-03,  5.8220e-03,  6.7100e-03,  7.6510e-03,
             8.6430e-03,  9.6840e-03,  1.0770e-02,  1.1900e-02,  1.3071e-02,
             1.4280e-02,  1.5523e-02,  1.6800e-02,  1.8106e-02,  1.9438e-02,
             2.0795e-02,  2.2173e-02,  2.3569e-02,  2.4981e-02,  2.6405e-02,
             2.7838e-02,  2.9279e-02,  3.0723e-02,  3.2168e-02,  3.3610e-02,
             3.5048e-02,  3.6478e-02,  3.7896e-02,  3.9300e-02,  4.0686e-02,
             4.2052e-02,  4.3394e-02,  4.4708e-02,  4.5992e-02,  4.7242e-02,
             4.8455e-02,  4.9626e-02,  5.0754e-02,  5.1833e-02,  5.2862e-02,
             5.3835e-02,  5.4749e-02,  5.5602e-02,  5.6390e-02,  5.7108e-02,
             5.7755e-02,  5.8326e-02,  5.8819e-02,  5.9230e-02,  5.9557e-02,
             5.9797e-02,  5.9947e-02,  6.0006e-02,  5.9971e-02,  5.9841e-02,
             5.9614e-02,  5.9288e-02,  5.8863e-02,  5.8338e-02,  5.7712e-02,
             5.6986e-02,  5.6159e-02,  5.5232e-02,  5.4206e-02,  5.3083e-02,
             5.1862e-02,  5.0546e-02,  4.9138e-02,  4.7638e-02,  4.6049e-02,
             4.4374e-02,  4.2615e-02,  4.0776e-02,  3.8859e-02,  3.6867e-02,
             3.4803e-02,  3.2671e-02,  3.0473e-02,  2.8213e-02,  2.5893e-02,
             2.3517e-02,  2.1088e-02,  1.8607e-02,  1.6078e-02,  1.3503e-02,
             1.0884e-02,  8.2230e-03,  5.5210e-03,  2.7790e-03,  0.0000e+00,
            -2.7790e-03, -5.5210e-03, -8.2230e-03, -1.0884e-02, -1.3503e-02,
            -1.6078e-02, -1.8607e-02, -2.1088e-02, -2.3517e-02, -2.5893e-02,
            -2.8213e-02, -3.0473e-02, -3.2671e-02, -3.4803e-02, -3.6867e-02,
            -3.8859e-02, -4.0776e-02, -4.2615e-02, -4.4374e-02, -4.6049e-02,
            -4.7638e-02, -4.9138e-02, -5.0546e-02, -5.1862e-02, -5.3083e-02,
            -5.4206e-02, -5.5232e-02, -5.6159e-02, -5.6986e-02, -5.7712e-02,
            -5.8338e-02, -5.8863e-02, -5.9288e-02, -5.9614e-02, -5.9841e-02,
            -5.9971e-02, -6.0006e-02, -5.9947e-02, -5.9797e-02, -5.9557e-02,
            -5.9230e-02, -5.8819e-02, -5.8326e-02, -5.7755e-02, -5.7108e-02,
            -5.6390e-02, -5.5602e-02, -5.4749e-02, -5.3835e-02, -5.2862e-02,
            -5.1833e-02, -5.0754e-02, -4.9626e-02, -4.8455e-02, -4.7242e-02,
            -4.5992e-02, -4.4708e-02, -4.3394e-02, -4.2052e-02, -4.0686e-02,
            -3.9300e-02, -3.7896e-02, -3.6478e-02, -3.5048e-02, -3.3610e-02,
            -3.2168e-02, -3.0723e-02, -2.9279e-02, -2.7838e-02, -2.6405e-02,
            -2.4981e-02, -2.3569e-02, -2.2173e-02, -2.0795e-02, -1.9438e-02,
            -1.8106e-02, -1.6800e-02, -1.5523e-02, -1.4280e-02, -1.3071e-02,
            -1.1900e-02, -1.0770e-02, -9.6840e-03, -8.6430e-03, -7.6510e-03,
            -6.7100e-03, -5.8220e-03, -4.9900e-03, -4.2160e-03, -3.5010e-03,
            -2.8490e-03, -2.2600e-03, -1.7370e-03, -1.2800e-03, -8.9100e-04,
            -5.7200e-04, -3.2200e-04, -1.4300e-04, -3.6000e-05,  0.0000e+00]])