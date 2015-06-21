# DistanceCalcLib
The DistanceCalcLib includes the frequently-used distance calculation methods.

**Implemeted distance calculation methods**

1. Euclidian Distance
2. Manhattan Distance
3. Dynamic Time Warp
4. Fréchet Distance

###Headers
Header Name   | Description
------------- | -------------
distanceCalc.h| All the distance calculation methods

###Usage

**distance_calculation**(*InputMatrix1, InputMatrix2, OutputMatrix, Row1, Row2, Col, Algorithm*)


###Arguments
 * InputMatrix1
 
 	Float pointer value. The pointer of the first input matrix. The matrix should be input as an 1-D float matrix.
 * InputMatrix2
 
 	Float pointer value. The pointer of the second input matrix. The matrix should be input as an 1-D float matrix.
 * OutputMatrix
 
 	Integar value. The pointer of the output matrix. The matrix would be outputed as an 1-D float matrix.
 * Row1
 
 	Integar value. The number of the first input matrix rows.
 * Row2
 
 	Integar value. The number of the second input matrix rows.
 * Col
 
 	Integar value. The number of the matrix columns.
 * Algorithm
 
 	Integar value. The index of the distance calculation algorithm. The index relationship is shown below:
	
	 Index Number  | Algorithm
	 ------------- | -------------
	 1|Euclidian Distance
	 2|manhattan Distance
	 3|Dynamic Time Warp
	 4|Fréchet Distance





