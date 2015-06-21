/*
 * distanceCalc.h
 *
 *  Created on: Jun 2, 2015
 *      Author: jackzhang
 */

#ifndef DISTANCECALC_H_
#define DISTANCECALC_H_

/*
 * Structure used to store the device information
 * With these information, we can optimize the GPU resources
 */


enum dAlgorithms{euclidian = 1, manhattan = 2};
void distance_calculation(float *in1, float *in2, float *out, int row1, int row2, int col, int algorithm);


#endif /* DISTANCECALC_H_ */
