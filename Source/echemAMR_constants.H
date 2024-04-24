#ifndef echemAMR_constants_H_
#define echemAMR_constants_H_

#define N_A 6.02214076e23
#define K_B 1.380649e-23
#define PI 3.14159265358979323846264338327950288
#define P_NTP 101325.0 //pressure at normal pressure and temperature
#define T_NTP 293.15  //temperature at normal pressure and temperature
#define FARADCONST 96485.332
#define GASCONST   8.31446261815324
#define ECHARGE    1.60217662e-19
#define EPS0       8.854187817e-12
#define FBYR FARADCONST/GASCONST

//these functions are to get x,y,z directions
//given a sweep direction
//these are used for viscous flux calculation
//if sweep direction is 0, normal-x, t1-y, t2-z
//if sweep direction is 1, normal-y, t1-z, t2-x
//if sweep direction is 2, normal-z, t1-x, t2-y
#define GET_XDIR(a) (a*(a-1)/2     - 2*a*(a-2))
#define GET_YDIR(a) ((a-1)*(a-2)/2 + a*(a-1))
#define GET_ZDIR(a) ((a-1)*(a-2)   - a*(a-2))

#endif
