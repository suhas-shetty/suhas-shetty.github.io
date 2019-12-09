/** WebGazer.js: Scalable Webcam EyeTracking Using User Interactions 
 * 
 * Copyright (c) 2016-2019, Brown HCI Group 

* Licensed under GPLv3. Companies with a valuation of less than $10M can use WebGazer.js under LGPLv3. 
*/

/**
 * Real-time object detector based on the Viola Jones Framework.
 * Compatible to OpenCV Haar Cascade Classifiers (stump based only).
 * 
 * Copyright (c) 2012, Martin Tschirsich

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
var objectdetect = (function() {
	"use strict";
    	
    var /**
		 * Converts from a 4-channel RGBA source image to a 1-channel grayscale
		 * image. Corresponds to the CV_RGB2GRAY OpenCV color space conversion.
		 * 
		 * @param {Array} src   4-channel 8-bit source image
		 * @param {Array} [dst] 1-channel 32-bit destination image
		 * 
		 * @return {Array} 1-channel 32-bit destination image
		 */
		convertRgbaToGrayscale = function(src, dst) {
			var srcLength = src.length;
			if (!dst) dst = new Uint32Array(srcLength >> 2);
			
			for (var i = 0; i < srcLength; i += 4) {
				dst[i >> 2] = (src[i] * 4899 + src[i + 1] * 9617 + src[i + 2] * 1868 + 8192) >> 14;
			}
			return dst;
		},
		
		/**
		 * Reduces the size of a given image by the given factor. Does NOT 
		 * perform interpolation. If interpolation is required, prefer using
		 * the drawImage() method of the <canvas> element.
		 * 
		 * @param {Array}  src       1-channel source image
		 * @param {Number} srcWidth	 Width of the source image
		 * @param {Number} srcHeight Height of the source image
		 * @param {Number} factor    Scaling down factor (> 1.0)
		 * @param {Array}  [dst]     1-channel destination image
		 * 
		 * @return {Array} 1-channel destination image
		 */
		rescaleImage = function(src, srcWidth, srcHeight, factor, dst) {
			var srcLength = srcHeight * srcWidth,
				dstWidth = ~~(srcWidth / factor),
				dstHeight = ~~(srcHeight / factor);
			
			if (!dst) dst = new src.constructor(dstWidth * srcHeight);
			
			for (var x = 0; x < dstWidth; ++x) {
				var dstIndex = x;
				for (var srcIndex = ~~(x * factor), srcEnd = srcIndex + srcLength; srcIndex < srcEnd; srcIndex += srcWidth) {
					dst[dstIndex] = src[srcIndex];
					dstIndex += dstWidth;
				}
			}
			
			var dstIndex = 0;
			for (var y = 0, yEnd = dstHeight * factor; y < yEnd; y += factor) {
				for (var srcIndex = ~~(y) * dstWidth, srcEnd=srcIndex + dstWidth; srcIndex < srcEnd; ++srcIndex) {
					dst[dstIndex] = dst[srcIndex];
					++dstIndex;
				}
			}
			return dst;
		},
		
		/**
		 * Horizontally mirrors a 1-channel source image.
		 * 
		 * @param {Array}  src       1-channel source image
		 * @param {Number} srcWidth  Width of the source image
		 * @param {Number} srcHeight Height of the source image
		 * @param {Array} [dst]      1-channel destination image
		 * 
		 * @return {Array} 1-channel destination image
		 */
		mirrorImage = function(src, srcWidth, srcHeight, dst) {
			if (!dst) dst = new src.constructor(src.length);
			
			var index = 0;
			for (var y = 0; y < srcHeight; ++y) {
				for (var x = (srcWidth >> 1); x >= 0; --x) {
					var swap = src[index + x];
					dst[index + x] = src[index + srcWidth - 1 - x];
					dst[index + srcWidth - 1 - x] = swap;
				}
				index += srcWidth;
			}
			return dst;
		},
		
		/**
		 * Computes the gradient magnitude using a sobel filter after
		 * applying gaussian smoothing (5x5 filter size). Useful for canny
		 * pruning.
		 * 
		 * @param {Array}  src      1-channel source image
		 * @param {Number} srcWidth Width of the source image
		 * @param {Number} srcWidth Height of the source image
		 * @param {Array}  [dst]    1-channel destination image
		 * 
		 * @return {Array} 1-channel destination image
		 */
		computeCanny = function(src, srcWidth, srcHeight, dst) {
			var srcLength = src.length;
			if (!dst) dst = new src.constructor(srcLength);
			var buffer1 = dst === src ? new src.constructor(srcLength) : dst;
			var buffer2 = new src.constructor(srcLength);
			
			// Gaussian filter with size=5, sigma=sqrt(2) horizontal pass:
			for (var x = 2; x < srcWidth - 2; ++x) {
				var index = x;
				for (var y = 0; y < srcHeight; ++y) {
					buffer1[index] =
						0.1117 * src[index - 2] +
						0.2365 * src[index - 1] +
						0.3036 * src[index] +
						0.2365 * src[index + 1] +
						0.1117 * src[index + 2];
					index += srcWidth;
				}
			}
			
			// Gaussian filter with size=5, sigma=sqrt(2) vertical pass:
			for (var x = 0; x < srcWidth; ++x) {
				var index = x + srcWidth;
				for (var y = 2; y < srcHeight - 2; ++y) {
					index += srcWidth;
					buffer2[index] =
						0.1117 * buffer1[index - srcWidth - srcWidth] +
						0.2365 * buffer1[index - srcWidth] +
						0.3036 * buffer1[index] +
						0.2365 * buffer1[index + srcWidth] +
						0.1117 * buffer1[index + srcWidth + srcWidth];
				}
			}
			
			// Compute gradient:
			var abs = Math.abs;
			for (var x = 2; x < srcWidth - 2; ++x) {
				var index = x + srcWidth;
				for (var y = 2; y < srcHeight - 2; ++y) {
					index += srcWidth;
					
					dst[index] = 
						abs(-     buffer2[index - 1 - srcWidth]
							+     buffer2[index + 1 - srcWidth]
							- 2 * buffer2[index - 1]
							+ 2 * buffer2[index + 1]
							-     buffer2[index - 1 + srcWidth]
							+     buffer2[index + 1 + srcWidth]) +
						
						abs(      buffer2[index - 1 - srcWidth]
							+ 2 * buffer2[index - srcWidth]
							+     buffer2[index + 1 - srcWidth]
							-     buffer2[index - 1 + srcWidth]
							- 2 * buffer2[index + srcWidth]
							-     buffer2[index + 1 + srcWidth]);
				}
			}
			return dst;
		},

		/**
		 * Computes the integral image of a 1-channel image. Arithmetic
		 * overflow may occur if the integral exceeds the limits for the
		 * destination image values ([0, 2^32-1] for an unsigned 32-bit image).
		 * The integral image is 1 pixel wider both in vertical and horizontal
		 * dimension compared to the source image.
		 * 
		 * SAT = Summed Area Table.
		 * 
		 * @param {Array}       src       1-channel source image
		 * @param {Number}      srcWidth  Width of the source image
		 * @param {Number}      srcHeight Height of the source image
		 * @param {Uint32Array} [dst]     1-channel destination image
		 * 
		 * @return {Uint32Array} 1-channel destination image
		 */
		computeSat = function(src, srcWidth, srcHeight, dst) {
			var dstWidth = srcWidth + 1;
			
			if (!dst) dst = new Uint32Array(srcWidth * srcHeight + dstWidth + srcHeight);
			
			for (var i = srcHeight * dstWidth; i >= 0; i -= dstWidth)
				dst[i] = 0;
			
			for (var x = 1; x <= srcWidth; ++x) {
				var column_sum = 0;
				var index = x;
				dst[x] = 0;
				
				for (var y = 1; y <= srcHeight; ++y) {
					column_sum += src[index - y];
					index += dstWidth;
					dst[index] = dst[index - 1] + column_sum;
				}
			}
			return dst;
		},
		
		/**
		 * Computes the squared integral image of a 1-channel image.
		 * @see computeSat()
		 * 
		 * @param {Array}       src       1-channel source image
		 * @param {Number}      srcWidth  Width of the source image
		 * @param {Number}      srcHeight Height of the source image
		 * @param {Uint32Array} [dst]     1-channel destination image
		 * 
		 * @return {Uint32Array} 1-channel destination image
		 */
		computeSquaredSat = function(src, srcWidth, srcHeight, dst) {
			var dstWidth = srcWidth + 1;
		
			if (!dst) dst = new Uint32Array(srcWidth * srcHeight + dstWidth + srcHeight);
			
			for (var i = srcHeight * dstWidth; i >= 0; i -= dstWidth)
				dst[i] = 0;
			
			for (var x = 1; x <= srcWidth; ++x) {
				var column_sum = 0;
				var index = x;
				dst[x] = 0;
				for (var y = 1; y <= srcHeight; ++y) {
					var val = src[index - y];
					column_sum += val * val;
					index += dstWidth;
					dst[index] = dst[index - 1] + column_sum;
				}
			}
			return dst;
		},
		
		/**
		 * Computes the rotated / tilted integral image of a 1-channel image.
		 * @see computeSat()
		 * 
		 * @param {Array}       src       1-channel source image
		 * @param {Number}      srcWidth  Width of the source image
		 * @param {Number}      srcHeight Height of the source image
		 * @param {Uint32Array} [dst]     1-channel destination image
		 * 
		 * @return {Uint32Array} 1-channel destination image
		 */
		computeRsat = function(src, srcWidth, srcHeight, dst) {
			var dstWidth = srcWidth + 1,
				srcHeightTimesDstWidth = srcHeight * dstWidth;
				
			if (!dst) dst = new Uint32Array(srcWidth * srcHeight + dstWidth + srcHeight);
			
			for (var i = srcHeightTimesDstWidth; i >= 0; i -= dstWidth)
				dst[i] = 0;
			
			for (var i = dstWidth - 1; i >= 0; --i)
				dst[i] = 0;
			
			var index = 0;
			for (var y = 0; y < srcHeight; ++y) {
				for (var x = 0; x < srcWidth; ++x) {
					dst[index + dstWidth + 1] = src[index - y] + dst[index];
					++index;
				}
				dst[index + dstWidth] += dst[index];
				index++;
			}
			
			for (var x = srcWidth - 1; x > 0; --x) {
				var index = x + srcHeightTimesDstWidth;
				for (var y = srcHeight; y > 0; --y) {
					index -= dstWidth;
					dst[index + dstWidth] += dst[index] + dst[index + 1];
				}
			}
			
			return dst;
		},
		
		/**
		 * Equalizes the histogram of an unsigned 1-channel image with integer 
		 * values in [0, 255]. Corresponds to the equalizeHist OpenCV function.
		 * 
		 * @param {Array}  src   1-channel integer source image
		 * @param {Number} step  Sampling stepsize, increase for performance
		 * @param {Array}  [dst] 1-channel destination image
		 * 
		 * @return {Array} 1-channel destination image
		 */
		equalizeHistogram = function(src, step, dst) {
			var srcLength = src.length;
			if (!dst) dst = src;
			if (!step) step = 5;
			
			// Compute histogram and histogram sum:
			var hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			            0, 0, 0, 0];
			
			for (var i = 0; i < srcLength; i += step) {
				++hist[src[i]];
			}
			
			// Compute integral histogram:
			var norm = 255 * step / srcLength,
				prev = 0;
			for (var i = 0; i < 256; ++i) {
				var h = hist[i];
				prev = h += prev;
				hist[i] = h * norm; // For non-integer src: ~~(h * norm + 0.5);
			}
			
			// Equalize image:
			for (var i = 0; i < srcLength; ++i) {
				dst[i] = hist[src[i]];
			}
			return dst;
		},
		
		/**
		 * Horizontally mirrors a cascase classifier. Useful to detect mirrored
		 * objects such as opposite hands.
		 * 
		 * @param {Array} dst Cascade classifier
		 * 
		 * @return {Array} Mirrored cascade classifier
		 */
		mirrorClassifier = function(src, dst) {
			if (!dst) dst = new src.constructor(src);
			var windowWidth  = src[0];
			
			for (var i = 1, iEnd = src.length - 1; i < iEnd; ) {
				++i;
				for (var j = 0, jEnd = src[++i]; j < jEnd; ++j) {
					if (src[++i]) {		
						// Simple classifier is tilted:
						for (var kEnd = i + src[++i] * 5; i < kEnd; ) {
							dst[i + 1] = windowWidth - src[i + 1];
							var width = src[i + 3];
							dst[i + 3] = src[i + 4];
							dst[i + 4] = width;
							i += 5;
						}
					} else {
						// Simple classifier is not tilted:
						for (var kEnd = i + src[++i] * 5; i < kEnd; ) {
							dst[i + 1] = windowWidth - src[i + 1] - src[i + 3];
							i += 5;
						}
					}
					i += 3;
				}
			}
			return dst;
		},
		
		/**
		 * Compiles a cascade classifier to be applicable to images
		 * of given dimensions. Speeds-up the actual detection process later on.
		 * 
		 * @param {Array}        src    Cascade classifier
		 * @param {Number}       width  Width of the source image
		 * @param {Number}       height Height of the source image
		 * @param {Float32Array} [dst]  Compiled cascade classifier
		 * 
		 * @return {Float32Array} Compiled cascade classifier
		 */
		compileClassifier = function(src, width, height, scale, dst) {
			width += 1;
			height += 1;
			if (!dst) dst = new Float32Array(src.length);
			var dstUint32 = new Uint32Array(dst.buffer);
			
			dstUint32[0] = src[0];
			dstUint32[1] = src[1];
			var dstIndex = 1;
			for (var srcIndex = 1, iEnd = src.length - 1; srcIndex < iEnd; ) {
				dst[++dstIndex] = src[++srcIndex];
				
				var numComplexClassifiers = dstUint32[++dstIndex] = src[++srcIndex];
				for (var j = 0, jEnd = numComplexClassifiers; j < jEnd; ++j) {
					
					var tilted = dst[++dstIndex] = src[++srcIndex];
					var numFeaturesTimes2 = dstUint32[++dstIndex] = src[++srcIndex] * 3;
					if (tilted) {
						for (var kEnd = dstIndex + numFeaturesTimes2; dstIndex < kEnd; ) {						
							var featureOffset = src[srcIndex + 1] + src[srcIndex + 2] * width,
								featureWidthTimesWidth = src[srcIndex + 3] * (width + 1),
								featureHeightTimesWidth = src[srcIndex + 4] * (width - 1);
							
							dstUint32[++dstIndex] = featureOffset;
							dstUint32[++dstIndex] = featureWidthTimesWidth + (featureHeightTimesWidth << 16);
							
							dst[++dstIndex] = src[srcIndex + 5];
							srcIndex += 5;
						}
					} else {
						for (var kEnd = dstIndex + numFeaturesTimes2; dstIndex < kEnd; ) {
							var featureOffset = src[srcIndex + 1] + src[srcIndex + 2] * width,
								featureWidth = src[srcIndex + 3],
								featureHeightTimesWidth = src[srcIndex + 4] * width;

							dstUint32[++dstIndex] = featureOffset;
							dstUint32[++dstIndex] = featureWidth + (featureHeightTimesWidth << 16);
							dst[++dstIndex] = src[srcIndex + 5];
							srcIndex += 5;
						}
					}
					var classifierThreshold = src[++srcIndex];
					for (var k = 0; k < numFeaturesTimes2;) {
						dst[dstIndex - k] /= classifierThreshold;
						k += 3;
					}

					if ((classifierThreshold < 0)) {
						dst[dstIndex + 2] = src[++srcIndex];
						dst[dstIndex + 1] = src[++srcIndex];
						dstIndex += 2;
					} else {
						dst[++dstIndex] = src[++srcIndex];
						dst[++dstIndex] = src[++srcIndex];
					}
				}
			}
			return dst.subarray(0, dstIndex+1);
		},
		
		/**
		 * Evaluates a compiled cascade classifier. Sliding window approach.
		 * 
		 * @param {Uint32Array}  sat        SAT of the source image
		 * @param {Uint32Array}  rsat       Rotated SAT of the source image
		 * @param {Uint32Array}  ssat       Squared SAT of the source image
		 * @param {Uint32Array}  [cannySat] SAT of the canny source image
		 * @param {Number}       width      Width of the source image
		 * @param {Number}       height     Height of the source image
		 * @param {Number}       step       Stepsize, increase for performance
		 * @param {Float32Array} classifier Compiled cascade classifier
		 * 
		 * @return {Array} Rectangles representing detected objects
		 */
		detect = function(sat, rsat, ssat, cannySat, width, height, step, classifier) {
			width  += 1;
			height += 1;
			
			var classifierUint32 = new Uint32Array(classifier.buffer),
				windowWidth  = classifierUint32[0],
				windowHeight = classifierUint32[1],
				windowHeightTimesWidth = windowHeight * width,
				area = windowWidth * windowHeight,
				inverseArea = 1 / area,
				widthTimesStep = width * step,
				rects = [];
			
			for (var x = 0; x + windowWidth < width; x += step) {
				var satIndex = x;
				for (var y = 0; y + windowHeight < height; y += step) {					
					var satIndex1 = satIndex + windowWidth,
						satIndex2 = satIndex + windowHeightTimesWidth,
						satIndex3 = satIndex2 + windowWidth,
						canny = false;

					// Canny test:
					if (cannySat) {
						var edgesDensity = (cannySat[satIndex] -
											cannySat[satIndex1] -
											cannySat[satIndex2] +
											cannySat[satIndex3]) * inverseArea;
						if (edgesDensity < 60 || edgesDensity > 200) {
							canny = true;
							satIndex += widthTimesStep;
							continue;
						}
					}
					
					// Normalize mean and variance of window area:
					var mean = (sat[satIndex] -
							    sat[satIndex1] -
						        sat[satIndex2] +
						        sat[satIndex3]),
						
						variance = (ssat[satIndex] -
						            ssat[satIndex1] -
						            ssat[satIndex2] +
						            ssat[satIndex3]) * area - mean * mean,
						            
						std = variance > 1 ? Math.sqrt(variance) : 1,
						found = true;
					
					// Evaluate cascade classifier aka 'stages':
					for (var i = 1, iEnd = classifier.length - 1; i < iEnd; ) {
						var complexClassifierThreshold = classifier[++i];
						// Evaluate complex classifiers aka 'trees':
						var complexClassifierSum = 0;
						for (var j = 0, jEnd = classifierUint32[++i]; j < jEnd; ++j) {
							
							// Evaluate simple classifiers aka 'nodes':
							var simpleClassifierSum = 0;
							
							if (classifierUint32[++i]) {
								// Simple classifier is tilted:
								for (var kEnd = i + classifierUint32[++i]; i < kEnd; ) {
									var f1 = satIndex + classifierUint32[++i],
										packed = classifierUint32[++i],
										f2 = f1 + (packed & 0xFFFF),
										f3 = f1 + (packed >> 16 & 0xFFFF);
									
									simpleClassifierSum += classifier[++i] *
										(rsat[f1] - rsat[f2] - rsat[f3] + rsat[f2 + f3 - f1]);
								}
							} else {
								// Simple classifier is not tilted:
								for (var kEnd = i + classifierUint32[++i]; i < kEnd; ) {
									var f1 = satIndex + classifierUint32[++i],
										packed = classifierUint32[++i],
										f2 = f1 + (packed & 0xFFFF),
										f3 = f1 + (packed >> 16 & 0xFFFF);
									
									simpleClassifierSum += classifier[++i] *
										(sat[f1] - sat[f2] - sat[f3] + sat[f2 + f3 - f1]);
								}
							}
							complexClassifierSum += classifier[i + (simpleClassifierSum > std ? 2 : 1)];
							i += 2;
						}
						if (complexClassifierSum < complexClassifierThreshold) {
							found = false;
							break;
						}
					}
					if (found) rects.push([x, y, windowWidth, windowHeight]);
					satIndex += widthTimesStep;
				}
			}
			return rects;
		},
	    
		/**
		 * Groups rectangles together using a rectilinear distance metric. For
		 * each group of related rectangles, a representative mean rectangle
		 * is returned.
		 * 
		 * @param {Array}  rects        Rectangles (Arrays of 4 floats)
		 * @param {Number} minNeighbors
		 *  
		 * @return {Array} Mean rectangles (Arrays of 4 floats)
		 */
		groupRectangles = function(rects, minNeighbors, confluence) {
			var rectsLength = rects.length;
			if (!confluence) confluence = 1.0;
			
	    	// Partition rects into similarity classes:
	    	var numClasses = 0;
	    	var labels = new Array(rectsLength);
			for (var i = 0; i < labels.length; ++i) {
				labels[i] = 0;
			}
			
			for (var i = 0; i < rectsLength; ++i) {
				var found = false;
				for (var j = 0; j < i; ++j) {
					
					// Determine similarity:
					var rect1 = rects[i];
					var rect2 = rects[j];
			        var delta = confluence * (Math.min(rect1[2], rect2[2]) + Math.min(rect1[3], rect2[3]));
			        if (Math.abs(rect1[0] - rect2[0]) <= delta &&
			        	Math.abs(rect1[1] - rect2[1]) <= delta &&
			        	Math.abs(rect1[0] + rect1[2] - rect2[0] - rect2[2]) <= delta &&
			        	Math.abs(rect1[1] + rect1[3] - rect2[1] - rect2[3]) <= delta) {
						
						labels[i] = labels[j];
						found = true;
						break;
					}
				}
				if (!found) {
					labels[i] = numClasses++;
				}
			}
			
			// Compute average rectangle (group) for each cluster:
			var groups = new Array(numClasses);
			
			for (var i = 0; i < numClasses; ++i) {
				groups[i] = [0, 0, 0, 0, 0];
			}
			
			for (var i = 0; i < rectsLength; ++i) {
				var rect = rects[i],
					group = groups[labels[i]];
				group[0] += rect[0];
				group[1] += rect[1];
				group[2] += rect[2];
				group[3] += rect[3];
				++group[4];
			}
			
			for (var i = numClasses - 1; i >= 0; --i) {
				var numNeighbors = groups[i][4];
				if (numNeighbors >= minNeighbors) {
					var group = groups[i];
					group[0] /= numNeighbors;
					group[1] /= numNeighbors;
					group[2] /= numNeighbors;
					group[3] /= numNeighbors;
				} else groups.splice(i, 1);
			}
			
			// Filter out small rectangles inside larger rectangles:
			var filteredGroups = [];
			for (var i = 0; i < numClasses; ++i) {
		        var r1 = groups[i];
		        
		        for (var j = 0; j < numClasses; ++j) {
		        	if (i === j) continue;
		            var r2 = groups[j];
		            var dx = r2[2] * 0.2;
		            var dy = r2[3] * 0.2;
		            
		            if (r1[0] >= r2[0] - dx &&
		                r1[1] >= r2[1] - dy &&
		                r1[0] + r1[2] <= r2[0] + r2[2] + dx &&
		                r1[1] + r1[3] <= r2[1] + r2[3] + dy) {
		            	break;
		            }
		        }
		        
		        if (j === numClasses) {
		        	filteredGroups.push(r1);
		        }
		    }
			return filteredGroups;
		};
	
	var detector = (function() {
		
		/**
		 * Creates a new detector - basically a convenient wrapper class around
		 * the js-objectdetect functions and hides away the technical details
		 * of multi-scale object detection on image, video or canvas elements.
		 * 
		 * @param width       Width of the detector
		 * @param height      Height of the detector
		 * @param scaleFactor Scaling factor for multi-scale detection
		 * @param classifier  Compiled cascade classifier
		 */
		function detector(width, height, scaleFactor, classifier) {
			this.canvas = document.createElement('canvas');
			this.canvas.width = width;
			this.canvas.height = height;
			this.context = this.canvas.getContext('2d');
			this.tilted = classifier.tilted;
			this.scaleFactor = scaleFactor;
			this.numScales = ~~(Math.log(Math.min(width / classifier[0], height / classifier[1])) / Math.log(scaleFactor));
			this.scaledGray = new Uint32Array(width * height);
			this.compiledClassifiers = [];
			var scale = 1;
			for (var i = 0; i < this.numScales; ++i) {
				var scaledWidth = ~~(width / scale);
				var scaledHeight = ~~(height / scale);
				this.compiledClassifiers[i] = objectdetect.compileClassifier(classifier, scaledWidth, scaledHeight);
				scale *= scaleFactor;
			}
		}
		
		/**
		 * Multi-scale object detection on image, video or canvas elements. 
		 * 
		 * @param image          HTML image, video or canvas element
		 * @param [group]        Detection results will be grouped by proximity
		 * @param [stepSize]     Increase for performance
		 * 
		 * @return Grouped rectangles
		 */
		detector.prototype.detect = function(image, group, stepSize) {
			if (!stepSize) stepSize = 1;
			
			var width = this.canvas.width;
			var height = this.canvas.height;
			
			this.context.drawImage(image, 0, 0, width, height);
			var imageData = this.context.getImageData(0, 0, width, height).data;
			this.gray = objectdetect.convertRgbaToGrayscale(imageData, this.gray);
			
			var rects = [];
			var scale = 1;
			for (var i = 0; i < this.numScales; ++i) {
				var scaledWidth = ~~(width / scale);
				var scaledHeight = ~~(height / scale);
				
				if (scale == 1) {
					this.scaledGray.set(this.gray);
				} else {
					this.scaledGray = objectdetect.rescaleImage(this.gray, width, height, scale, this.scaledGray);
				}
				
				this.sat = objectdetect.computeSat(this.scaledGray, scaledWidth, scaledHeight, this.sat);
				this.ssat = objectdetect.computeSquaredSat(this.scaledGray, scaledWidth, scaledHeight, this.ssat);
				if (this.tilted) this.rsat = objectdetect.computeRsat(this.scaledGray, scaledWidth, scaledHeight, this.rsat);
				this.cannysat = undefined;

				var newRects = detect(this.sat, this.rsat, this.ssat, this.cannysat, scaledWidth, scaledHeight, stepSize, this.compiledClassifiers[i]);
				for (var j = newRects.length - 1; j >= 0; --j) {
					newRects[j][0] *= width / scaledWidth;
					newRects[j][1] *= height / scaledHeight;
					newRects[j][2] *= width / scaledWidth;
					newRects[j][3] *= height / scaledHeight;
				}
				rects = rects.concat(newRects);
				
				scale *= this.scaleFactor;
			}				
			return (group ? objectdetect.groupRectangles(rects, group) : rects).sort(function(r1, r2) {return r2[4] - r1[4];});
		};
		
		return detector;
	})();
	
	return {
		convertRgbaToGrayscale: convertRgbaToGrayscale,
		rescaleImage: rescaleImage,
		mirrorImage: mirrorImage,
		computeCanny: computeCanny,
		equalizeHistogram: equalizeHistogram,
		computeSat: computeSat,
		computeRsat: computeRsat,
		computeSquaredSat: computeSquaredSat,
		mirrorClassifier: mirrorClassifier,
		compileClassifier: compileClassifier,
		detect: detect,
		groupRectangles: groupRectangles,
		detector: detector
	};
})();
(function(module) {
    "use strict";
    
    var classifier = [20,20,-1.4562760591506958,6,0,2,0,8,20,12,-1.,0,14,20,6,2.,0.1296395957469940,-0.7730420827865601,0.6835014820098877,0,2,9,1,4,15,-1.,9,6,4,5,3.,-0.0463268086314201,0.5735275149345398,-0.4909768998622894,0,2,6,10,9,2,-1.,9,10,3,2,3.,-0.0161730907857418,0.6025434136390686,-0.3161070942878723,0,2,7,0,10,9,-1.,7,3,10,3,3.,-0.0458288416266441,0.6417754888534546,-0.1554504036903381,0,2,12,2,2,18,-1.,12,8,2,6,3.,-0.0537596195936203,0.5421931743621826,-0.2048082947731018,0,2,8,6,8,6,-1.,8,9,8,3,2.,0.0341711901128292,-0.2338819056749344,0.4841090142726898,-1.2550230026245117,12,0,2,2,0,17,18,-1.,2,6,17,6,3.,-0.2172762006521225,0.7109889984130859,-0.5936073064804077,0,2,10,10,1,8,-1.,10,14,1,4,2.,0.0120719699189067,-0.2824048101902008,0.5901355147361755,0,2,7,10,9,2,-1.,10,10,3,2,3.,-0.0178541392087936,0.5313752293586731,-0.2275896072387695,0,2,5,1,6,6,-1.,5,3,6,2,3.,0.0223336108028889,-0.1755609959363937,0.6335613727569580,0,2,3,1,15,9,-1.,3,4,15,3,3.,-0.0914200171828270,0.6156309247016907,-0.1689953058958054,0,2,6,3,9,6,-1.,6,5,9,2,3.,0.0289736501872540,-0.1225007995963097,0.7440117001533508,0,2,8,17,6,3,-1.,10,17,2,3,3.,7.8203463926911354e-003,0.1697437018156052,-0.6544165015220642,0,2,9,10,9,1,-1.,12,10,3,1,3.,0.0203404892235994,-0.1255664974451065,0.8271045088768005,0,2,1,7,6,11,-1.,3,7,2,11,3.,-0.0119261499494314,0.3860568106174469,-0.2099234014749527,0,2,9,18,3,1,-1.,10,18,1,1,3.,-9.7281101625412703e-004,-0.6376119256019592,0.1295239031314850,0,2,16,16,1,2,-1.,16,17,1,1,2.,1.8322050891583785e-005,-0.3463147878646851,0.2292426973581314,0,2,9,17,6,3,-1.,11,17,2,3,3.,-8.0854417756199837e-003,-0.6366580128669739,0.1307865977287293,-1.3728189468383789,9,0,2,8,0,5,18,-1.,8,6,5,6,3.,-0.1181226968765259,0.6784452199935913,-0.5004578232765198,0,2,6,7,9,7,-1.,9,7,3,7,3.,-0.0343327596783638,0.6718636155128479,-0.3574487864971161,0,2,14,6,6,10,-1.,16,6,2,10,3.,-0.0215307995676994,0.7222070097923279,-0.1819241940975189,0,2,9,8,9,5,-1.,12,8,3,5,3.,-0.0219099707901478,0.6652938723564148,-0.2751022875308991,0,2,3,7,9,6,-1.,6,7,3,6,3.,-0.0287135392427444,0.6995570063591003,-0.1961558014154434,0,2,1,7,6,6,-1.,3,7,2,6,3.,-0.0114674801006913,0.5926734805107117,-0.2209735065698624,0,2,16,0,4,18,-1.,16,6,4,6,3.,-0.0226111691445112,0.3448306918144226,-0.3837955892086029,0,2,0,17,3,3,-1.,0,18,3,1,3.,-1.9308089977130294e-003,-0.7944571971893311,0.1562865972518921,0,2,16,0,2,1,-1.,17,0,1,1,2.,5.6419910833938047e-005,-0.3089601099491119,0.3543108999729157,-1.2879480123519897,16,0,2,0,8,20,12,-1.,0,14,20,6,2.,0.1988652050495148,-0.5286070108413696,0.3553672134876251,0,2,6,6,9,8,-1.,9,6,3,8,3.,-0.0360089391469955,0.4210968911647797,-0.3934898078441620,0,2,5,3,12,9,-1.,5,6,12,3,3.,-0.0775698497891426,0.4799154102802277,-0.2512216866016388,0,2,4,16,1,2,-1.,4,17,1,1,2.,8.2630853285081685e-005,-0.3847548961639404,0.3184922039508820,0,2,18,10,2,1,-1.,19,10,1,1,2.,3.2773229759186506e-004,-0.2642731964588165,0.3254724144935608,0,2,9,8,6,5,-1.,11,8,2,5,3.,-0.0185748506337404,0.4673658907413483,-0.1506727039813995,0,2,0,0,2,1,-1.,1,0,1,1,2.,-7.0008762122597545e-005,0.2931315004825592,-0.2536509931087494,0,2,6,8,6,6,-1.,8,8,2,6,3.,-0.0185521300882101,0.4627366065979004,-0.1314805001020432,0,2,11,7,6,7,-1.,13,7,2,7,3.,-0.0130304200574756,0.4162721931934357,-0.1775148957967758,0,2,19,14,1,2,-1.,19,15,1,1,2.,6.5694141085259616e-005,-0.2803510129451752,0.2668074071407318,0,2,6,17,1,2,-1.,6,18,1,1,2.,1.7005260451696813e-004,-0.2702724933624268,0.2398165017366409,0,2,14,7,2,7,-1.,15,7,1,7,2.,-3.3129199873656034e-003,0.4441143870353699,-0.1442888975143433,0,2,6,8,2,4,-1.,7,8,1,4,2.,1.7583490116521716e-003,-0.1612619012594223,0.4294076859951019,0,2,5,8,12,6,-1.,5,10,12,2,3.,-0.0251947492361069,0.4068729877471924,-0.1820258051156998,0,2,2,17,1,3,-1.,2,18,1,1,3.,1.4031709870323539e-003,0.0847597867250443,-0.8001856803894043,0,2,6,7,3,6,-1.,7,7,1,6,3.,-7.3991729877889156e-003,0.5576609969139099,-0.1184315979480743,-1.2179850339889526,23,0,2,6,7,9,12,-1.,9,7,3,12,3.,-0.0299430806189775,0.3581081032752991,-0.3848763108253479,0,2,6,2,11,12,-1.,6,6,11,4,3.,-0.1256738007068634,0.3931693136692047,-0.3001225888729096,0,2,1,12,5,8,-1.,1,16,5,4,2.,5.3635272197425365e-003,-0.4390861988067627,0.1925701051950455,0,2,14,7,6,7,-1.,16,7,2,7,3.,-8.0971820279955864e-003,0.3990666866302490,-0.2340787053108215,0,2,10,8,6,6,-1.,12,8,2,6,3.,-0.0165979098528624,0.4209528863430023,-0.2267484068870544,0,2,16,18,4,2,-1.,16,19,4,1,2.,-2.0199299324303865e-003,-0.7415673136711121,0.1260118931531906,0,2,18,17,2,3,-1.,18,18,2,1,3.,-1.5202340437099338e-003,-0.7615460157394409,0.0863736122846603,0,2,9,7,3,7,-1.,10,7,1,7,3.,-4.9663940444588661e-003,0.4218223989009857,-0.1790491938591003,0,2,5,6,6,8,-1.,7,6,2,8,3.,-0.0192076005041599,0.4689489901065826,-0.1437875032424927,0,2,2,6,6,11,-1.,4,6,2,11,3.,-0.0122226802632213,0.3284207880496979,-0.2180214971303940,0,2,8,10,12,8,-1.,8,14,12,4,2.,0.0575486682355404,-0.3676880896091461,0.2435711026191711,0,2,7,17,6,3,-1.,9,17,2,3,3.,-9.5794079825282097e-003,-0.7224506735801697,0.0636645630002022,0,2,10,9,3,3,-1.,11,9,1,3,3.,-2.9545740690082312e-003,0.3584643900394440,-0.1669632941484451,0,2,8,8,3,6,-1.,9,8,1,6,3.,-4.2017991654574871e-003,0.3909480869770050,-0.1204179003834724,0,2,7,0,6,5,-1.,9,0,2,5,3.,-0.0136249903589487,-0.5876771807670593,0.0884047299623489,0,2,6,17,1,3,-1.,6,18,1,1,3.,6.2853112467564642e-005,-0.2634845972061157,0.2141927927732468,0,2,0,18,4,2,-1.,0,19,4,1,2.,-2.6782939676195383e-003,-0.7839016914367676,0.0805269628763199,0,2,4,1,11,9,-1.,4,4,11,3,3.,-0.0705971792340279,0.4146926105022430,-0.1398995965719223,0,2,3,1,14,9,-1.,3,4,14,3,3.,0.0920936465263367,-0.1305518001317978,0.5043578147888184,0,2,0,9,6,4,-1.,2,9,2,4,3.,-8.8004386052489281e-003,0.3660975098609924,-0.1403664946556091,0,2,18,13,1,2,-1.,18,14,1,1,2.,7.5080977694597095e-005,-0.2970443964004517,0.2070294022560120,0,2,13,5,3,11,-1.,14,5,1,11,3.,-2.9870450962334871e-003,0.3561570048332214,-0.1544596999883652,0,3,0,18,8,2,-1.,0,18,4,1,2.,4,19,4,1,2.,-2.6441509835422039e-003,-0.5435351729393005,0.1029511019587517,-1.2905240058898926,27,0,2,5,8,12,5,-1.,9,8,4,5,3.,-0.0478624701499939,0.4152823984622955,-0.3418582081794739,0,2,4,7,11,10,-1.,4,12,11,5,2.,0.0873505324125290,-0.3874978125095367,0.2420420050621033,0,2,14,9,6,4,-1.,16,9,2,4,3.,-0.0168494991958141,0.5308247804641724,-0.1728291064500809,0,2,0,7,6,8,-1.,3,7,3,8,2.,-0.0288700293749571,0.3584350943565369,-0.2240259051322937,0,2,0,16,3,3,-1.,0,17,3,1,3.,2.5679389946162701e-003,0.1499049961566925,-0.6560940742492676,0,2,7,11,12,1,-1.,11,11,4,1,3.,-0.0241166595369577,0.5588967800140381,-0.1481028050184250,0,2,4,8,9,4,-1.,7,8,3,4,3.,-0.0328266583383083,0.4646868109703064,-0.1078552976250649,0,2,5,16,6,4,-1.,7,16,2,4,3.,-0.0152330603450537,-0.7395442724227905,0.0562368817627430,0,2,18,17,1,3,-1.,18,18,1,1,3.,-3.0209511169232428e-004,-0.4554882049560547,0.0970698371529579,0,2,18,17,1,3,-1.,18,18,1,1,3.,7.5365108205005527e-004,0.0951472967863083,-0.5489501953125000,0,3,4,9,4,10,-1.,4,9,2,5,2.,6,14,2,5,2.,-0.0106389503926039,0.4091297090053558,-0.1230840981006622,0,2,4,8,6,4,-1.,6,8,2,4,3.,-7.5217830017209053e-003,0.4028914868831635,-0.1604878008365631,0,2,10,2,2,18,-1.,10,8,2,6,3.,-0.1067709997296333,0.6175932288169861,-0.0730911865830421,0,3,0,5,8,6,-1.,0,5,4,3,2.,4,8,4,3,2.,0.0162569191306829,-0.1310368031263351,0.3745365142822266,0,2,6,0,6,5,-1.,8,0,2,5,3.,-0.0206793602555990,-0.7140290737152100,0.0523900091648102,0,2,18,0,2,14,-1.,18,7,2,7,2.,0.0170523691922426,0.1282286047935486,-0.3108068108558655,0,2,8,18,4,2,-1.,10,18,2,2,2.,-5.7122060097754002e-003,-0.6055650711059570,0.0818847566843033,0,2,1,17,6,3,-1.,1,18,6,1,3.,2.0851430235779844e-005,-0.2681298851966858,0.1445384025573731,0,2,11,8,3,5,-1.,12,8,1,5,3.,7.9284431412816048e-003,-0.0787953510880470,0.5676258206367493,0,2,11,8,3,4,-1.,12,8,1,4,3.,-2.5217379443347454e-003,0.3706862926483154,-0.1362057030200958,0,2,11,0,6,5,-1.,13,0,2,5,3.,-0.0224261991679668,-0.6870499849319458,0.0510628595948219,0,2,1,7,6,7,-1.,3,7,2,7,3.,-7.6451441273093224e-003,0.2349222004413605,-0.1790595948696137,0,2,0,13,1,3,-1.,0,14,1,1,3.,-1.1175329564139247e-003,-0.5986905097961426,0.0743244364857674,0,2,3,2,9,6,-1.,3,4,9,2,3.,0.0192127898335457,-0.1570255011320114,0.2973746955394745,0,2,8,6,9,2,-1.,8,7,9,1,2.,5.6293429806828499e-003,-0.0997690185904503,0.4213027060031891,0,2,0,14,3,6,-1.,0,16,3,2,3.,-9.5671862363815308e-003,-0.6085879802703857,0.0735062584280968,0,2,1,11,6,4,-1.,3,11,2,4,3.,0.0112179601565003,-0.1032081022858620,0.4190984964370728,-1.1600480079650879,28,0,2,6,9,9,3,-1.,9,9,3,3,3.,-0.0174864400178194,0.3130728006362915,-0.3368118107318878,0,2,6,0,9,6,-1.,6,2,9,2,3.,0.0307146497070789,-0.1876619011163712,0.5378080010414124,0,2,8,5,6,6,-1.,8,7,6,2,3.,-0.0221887193620205,0.3663788139820099,-0.1612481027841568,0,2,1,12,2,1,-1.,2,12,1,1,2.,-5.0700771680567414e-005,0.2124571055173874,-0.2844462096691132,0,2,10,10,6,2,-1.,12,10,2,2,3.,-7.0170420221984386e-003,0.3954311013221741,-0.1317359060049057,0,2,13,8,6,6,-1.,15,8,2,6,3.,-6.8563609384000301e-003,0.3037385940551758,-0.2065781950950623,0,2,6,16,6,4,-1.,8,16,2,4,3.,-0.0141292596235871,-0.7650300860404968,0.0982131883502007,0,2,8,0,9,9,-1.,8,3,9,3,3.,-0.0479154810309410,0.4830738902091980,-0.1300680935382843,0,2,18,17,1,3,-1.,18,18,1,1,3.,4.7032979637151584e-005,-0.2521657049655914,0.2438668012619019,0,2,18,17,1,3,-1.,18,18,1,1,3.,1.0221180273219943e-003,0.0688576027750969,-0.6586114168167114,0,2,7,10,3,3,-1.,8,10,1,3,3.,-2.6056109927594662e-003,0.4294202923774719,-0.1302246004343033,0,3,9,14,2,2,-1.,9,14,1,1,2.,10,15,1,1,2.,5.4505340813193470e-005,-0.1928862035274506,0.2895849943161011,0,3,9,14,2,2,-1.,9,14,1,1,2.,10,15,1,1,2.,-6.6721157054416835e-005,0.3029071092605591,-0.1985436975955963,0,2,0,8,19,12,-1.,0,14,19,6,2.,0.2628143131732941,-0.2329394072294235,0.2369246035814285,0,2,7,6,9,14,-1.,10,6,3,14,3.,-0.0235696695744991,0.1940104067325592,-0.2848461866378784,0,2,13,8,3,4,-1.,14,8,1,4,3.,-3.9120172150433064e-003,0.5537897944450378,-0.0956656783819199,0,2,4,17,1,3,-1.,4,18,1,1,3.,5.0788799853762612e-005,-0.2391265928745270,0.2179948985576630,0,2,4,9,6,3,-1.,6,9,2,3,3.,-7.8732017427682877e-003,0.4069742858409882,-0.1276804059743881,0,2,2,18,5,2,-1.,2,19,5,1,2.,-1.6778609715402126e-003,-0.5774465799331665,0.0973247885704041,0,3,7,8,2,2,-1.,7,8,1,1,2.,8,9,1,1,2.,-2.6832430739887059e-004,0.2902188003063202,-0.1683126986026764,0,3,7,8,2,2,-1.,7,8,1,1,2.,8,9,1,1,2.,7.8687182394787669e-005,-0.1955157071352005,0.2772096991539002,0,2,5,10,13,2,-1.,5,11,13,1,2.,0.0129535002633929,-0.0968383178114891,0.4032387137413025,0,2,10,8,1,9,-1.,10,11,1,3,3.,-0.0130439596250653,0.4719856977462769,-0.0892875492572784,0,3,15,8,2,12,-1.,15,8,1,6,2.,16,14,1,6,2.,3.0261781066656113e-003,-0.1362338066101074,0.3068627119064331,0,2,4,0,3,5,-1.,5,0,1,5,3.,-6.0438038781285286e-003,-0.7795410156250000,0.0573163107037544,0,2,12,6,3,7,-1.,13,6,1,7,3.,-2.2507249377667904e-003,0.3087705969810486,-0.1500630974769592,0,2,7,16,6,4,-1.,9,16,2,4,3.,0.0158268101513386,0.0645518898963928,-0.7245556712150574,0,2,9,16,2,1,-1.,10,16,1,1,2.,6.5864507632795721e-005,-0.1759884059429169,0.2321038991212845,-1.2257250547409058,36,0,2,6,10,9,2,-1.,9,10,3,2,3.,-0.0278548691421747,0.4551844894886017,-0.1809991002082825,0,2,0,6,15,14,-1.,0,13,15,7,2.,0.1289504021406174,-0.5256553292274475,0.1618890017271042,0,2,9,1,5,6,-1.,9,3,5,2,3.,0.0244031809270382,-0.1497496068477631,0.4235737919807434,0,2,3,9,3,4,-1.,4,9,1,4,3.,-2.4458570405840874e-003,0.3294866979122162,-0.1744769066572189,0,2,5,7,3,6,-1.,6,7,1,6,3.,-3.5336529836058617e-003,0.4742664098739624,-0.0736183598637581,0,2,17,16,1,2,-1.,17,17,1,1,2.,5.1358150813030079e-005,-0.3042193055152893,0.1563327014446259,0,2,9,8,6,12,-1.,11,8,2,12,3.,-0.0162256807088852,0.2300218045711517,-0.2035982012748718,0,2,6,10,6,1,-1.,8,10,2,1,3.,-4.6007009223103523e-003,0.4045926928520203,-0.1348544061183929,0,2,7,17,9,3,-1.,10,17,3,3,3.,-0.0219289995729923,-0.6872448921203613,0.0806842669844627,0,2,14,18,6,2,-1.,14,19,6,1,2.,-2.8971210122108459e-003,-0.6961960792541504,0.0485452190041542,0,2,9,5,3,14,-1.,10,5,1,14,3.,-4.4074649922549725e-003,0.2516626119613648,-0.1623664945363998,0,2,8,16,9,4,-1.,11,16,3,4,3.,0.0284371692687273,0.0603942610323429,-0.6674445867538452,0,2,0,0,4,14,-1.,0,7,4,7,2.,0.0832128822803497,0.0643579214811325,-0.5362604260444641,0,2,8,1,6,3,-1.,10,1,2,3,3.,-0.0124193299561739,-0.7081686258316040,0.0575266107916832,0,2,6,8,3,4,-1.,7,8,1,4,3.,-4.6992599964141846e-003,0.5125433206558228,-0.0873508006334305,0,2,4,8,3,4,-1.,5,8,1,4,3.,-7.8025809489190578e-004,0.2668766081333160,-0.1796150952577591,0,2,5,1,6,5,-1.,7,1,2,5,3.,-0.0197243392467499,-0.6756373047828674,0.0729419067502022,0,2,1,18,1,2,-1.,1,19,1,1,2.,1.0269250487908721e-003,0.0539193190634251,-0.5554018020629883,0,2,7,0,6,6,-1.,7,2,6,2,3.,-0.0259571895003319,0.5636252760887146,-0.0718983933329582,0,2,0,18,4,2,-1.,0,19,4,1,2.,-1.2552699772641063e-003,-0.5034663081169128,0.0896914526820183,0,2,12,3,8,12,-1.,12,7,8,4,3.,-0.0499705784022808,0.1768511980772018,-0.2230195999145508,0,2,12,9,3,4,-1.,13,9,1,4,3.,-2.9899610672146082e-003,0.3912242054939270,-0.1014975011348724,0,2,12,8,3,5,-1.,13,8,1,5,3.,4.8546842299401760e-003,-0.1177017986774445,0.4219093918800354,0,2,16,0,2,1,-1.,17,0,1,1,2.,1.0448860120959580e-004,-0.1733397990465164,0.2234444022178650,0,2,5,17,1,3,-1.,5,18,1,1,3.,5.9689260524464771e-005,-0.2340963035821915,0.1655824035406113,0,2,10,2,3,6,-1.,10,4,3,2,3.,-0.0134239196777344,0.4302381873130798,-0.0997236520051956,0,2,4,17,2,3,-1.,4,18,2,1,3.,2.2581999655812979e-003,0.0727209895849228,-0.5750101804733276,0,2,12,7,1,9,-1.,12,10,1,3,3.,-0.0125462803989649,0.3618457913398743,-0.1145701035857201,0,2,7,6,3,9,-1.,8,6,1,9,3.,-2.8705769218504429e-003,0.2821053862571716,-0.1236755028367043,0,2,17,13,3,6,-1.,17,15,3,2,3.,0.0197856407612562,0.0478767491877079,-0.8066623806953430,0,2,7,7,3,8,-1.,8,7,1,8,3.,4.7588930465281010e-003,-0.1092538982629776,0.3374697864055634,0,2,5,0,3,5,-1.,6,0,1,5,3.,-6.9974269717931747e-003,-0.8029593825340271,0.0457067005336285,0,2,4,6,9,8,-1.,7,6,3,8,3.,-0.0130334803834558,0.1868043988943100,-0.1768891066312790,0,2,2,9,3,3,-1.,3,9,1,3,3.,-1.3742579612880945e-003,0.2772547900676727,-0.1280900985002518,0,2,16,18,4,2,-1.,16,19,4,1,2.,2.7657810132950544e-003,0.0907589420676231,-0.4259473979473114,0,2,17,10,3,10,-1.,17,15,3,5,2.,2.8941841446794569e-004,-0.3881632983684540,0.0892677977681160,-1.2863140106201172,47,0,2,8,9,6,4,-1.,10,9,2,4,3.,-0.0144692296162248,0.3750782907009125,-0.2492828965187073,0,2,5,2,10,12,-1.,5,6,10,4,3.,-0.1331762969493866,0.3016637861728668,-0.2241407036781311,0,2,6,9,6,3,-1.,8,9,2,3,3.,-0.0101321600377560,0.3698559105396271,-0.1785001009702683,0,2,11,7,3,7,-1.,12,7,1,7,3.,-7.8511182218790054e-003,0.4608676135540009,-0.1293139010667801,0,2,12,8,6,4,-1.,14,8,2,4,3.,-0.0142958397045732,0.4484142959117889,-0.1022624000906944,0,2,14,8,6,5,-1.,16,8,2,5,3.,-5.9606940485537052e-003,0.2792798876762390,-0.1532382965087891,0,2,12,12,2,4,-1.,12,14,2,2,2.,0.0109327696263790,-0.1514174044132233,0.3988964855670929,0,2,3,15,1,2,-1.,3,16,1,1,2.,5.0430990086169913e-005,-0.2268157005310059,0.2164438962936401,0,2,12,7,3,4,-1.,13,7,1,4,3.,-5.8431681245565414e-003,0.4542014896869659,-0.1258715987205505,0,2,10,0,6,6,-1.,12,0,2,6,3.,-0.0223462097346783,-0.6269019246101379,0.0824031233787537,0,2,10,6,3,8,-1.,11,6,1,8,3.,-4.8836669884622097e-003,0.2635925114154816,-0.1468663066625595,0,2,16,17,1,2,-1.,16,18,1,1,2.,7.5506002758629620e-005,-0.2450702041387558,0.1667888015508652,0,2,16,16,1,3,-1.,16,17,1,1,3.,-4.9026997294276953e-004,-0.4264996051788330,0.0899735614657402,0,2,11,11,1,2,-1.,11,12,1,1,2.,1.4861579984426498e-003,-0.1204025000333786,0.3009765148162842,0,2,3,7,6,9,-1.,5,7,2,9,3.,-0.0119883399456739,0.2785247862339020,-0.1224434003233910,0,2,4,18,9,1,-1.,7,18,3,1,3.,0.0105022396892309,0.0404527597129345,-0.7405040860176086,0,2,0,11,4,9,-1.,0,14,4,3,3.,-0.0309630092233419,-0.6284269094467163,0.0480137616395950,0,2,9,17,6,3,-1.,11,17,2,3,3.,0.0114145204424858,0.0394052118062973,-0.7167412042617798,0,2,7,8,6,12,-1.,9,8,2,12,3.,-0.0123370001092553,0.1994132995605469,-0.1927430033683777,0,2,6,8,3,4,-1.,7,8,1,4,3.,-5.9942267835140228e-003,0.5131816267967224,-0.0616580583155155,0,2,3,17,1,3,-1.,3,18,1,1,3.,-1.1923230485990644e-003,-0.7260529994964600,0.0506527200341225,0,2,11,9,6,4,-1.,13,9,2,4,3.,-7.4582789093255997e-003,0.2960307896137238,-0.1175478994846344,0,2,6,1,3,2,-1.,7,1,1,2,3.,2.7877509128302336e-003,0.0450687110424042,-0.6953541040420532,0,2,1,0,2,1,-1.,2,0,1,1,2.,-2.2503209766000509e-004,0.2004725039005280,-0.1577524989843369,0,3,1,0,2,14,-1.,1,0,1,7,2.,2,7,1,7,2.,-5.0367889925837517e-003,0.2929981946945190,-0.1170049980282784,0,2,5,5,11,8,-1.,5,9,11,4,2.,0.0747421607375145,-0.1139231994748116,0.3025662004947662,0,2,9,3,5,6,-1.,9,5,5,2,3.,0.0202555190771818,-0.1051589027047157,0.4067046046257019,0,2,7,9,5,10,-1.,7,14,5,5,2.,0.0442145094275475,-0.2763164043426514,0.1236386969685555,0,2,15,10,2,2,-1.,16,10,1,2,2.,-8.7259558495134115e-004,0.2435503005981445,-0.1330094933509827,0,2,0,18,8,2,-1.,0,19,8,1,2.,-2.4453739169985056e-003,-0.5386617183685303,0.0625106468796730,0,2,7,17,1,3,-1.,7,18,1,1,3.,8.2725353422574699e-005,-0.2077220976352692,0.1627043932676315,0,2,7,2,11,6,-1.,7,4,11,2,3.,-0.0366271100938320,0.3656840920448303,-0.0903302803635597,0,2,8,3,9,3,-1.,8,4,9,1,3.,3.0996399000287056e-003,-0.1318302005529404,0.2535429894924164,0,2,0,9,2,2,-1.,0,10,2,1,2.,-2.4709280114620924e-003,-0.5685349702835083,0.0535054318606853,0,2,0,5,3,6,-1.,0,7,3,2,3.,-0.0141146704554558,-0.4859901070594788,0.0584852509200573,0,3,6,7,2,2,-1.,6,7,1,1,2.,7,8,1,1,2.,8.4537261864170432e-004,-0.0800936371088028,0.4026564955711365,0,2,7,6,3,6,-1.,8,6,1,6,3.,-7.1098632179200649e-003,0.4470323920249939,-0.0629474371671677,0,2,12,1,6,4,-1.,14,1,2,4,3.,-0.0191259607672691,-0.6642286777496338,0.0498227700591087,0,2,9,11,6,8,-1.,11,11,2,8,3.,-5.0773010589182377e-003,0.1737940013408661,-0.1685059964656830,0,2,17,15,3,3,-1.,17,16,3,1,3.,-2.9198289848864079e-003,-0.6011028289794922,0.0574279390275478,0,2,6,6,3,9,-1.,6,9,3,3,3.,-0.0249021500349045,0.2339798063039780,-0.1181845963001251,0,3,0,5,8,6,-1.,0,5,4,3,2.,4,8,4,3,2.,0.0201477799564600,-0.0894598215818405,0.3602440059185028,0,2,0,6,1,3,-1.,0,7,1,1,3.,1.7597640398889780e-003,0.0494584403932095,-0.6310262084007263,0,2,17,0,2,6,-1.,18,0,1,6,2.,1.3812039978802204e-003,-0.1521805971860886,0.1897173970937729,0,2,10,17,6,3,-1.,12,17,2,3,3.,-0.0109045403078198,-0.5809738039970398,0.0448627285659313,0,3,13,15,2,2,-1.,13,15,1,1,2.,14,16,1,1,2.,7.5157178798690438e-005,-0.1377734988927841,0.1954316049814224,0,2,4,0,12,3,-1.,4,1,12,1,3.,3.8649770431220531e-003,-0.1030222997069359,0.2537496984004974,-1.1189440488815308,48,0,2,5,3,10,9,-1.,5,6,10,3,3.,-0.1021588966250420,0.4168125987052918,-0.1665562987327576,0,2,7,7,9,7,-1.,10,7,3,7,3.,-0.0519398190081120,0.3302395045757294,-0.2071571052074432,0,2,5,8,9,6,-1.,8,8,3,6,3.,-0.0427177809178829,0.2609373033046722,-0.1601389050483704,0,2,0,16,6,2,-1.,0,17,6,1,2.,4.3890418601222336e-004,-0.3475053012371063,0.1391891986131668,0,2,12,6,7,14,-1.,12,13,7,7,2.,0.0242643896490335,-0.4255205988883972,0.1357838064432144,0,2,13,7,6,8,-1.,15,7,2,8,3.,-0.0238205995410681,0.3174980878829956,-0.1665204018354416,0,2,2,10,6,3,-1.,4,10,2,3,3.,-7.0518180727958679e-003,0.3094717860221863,-0.1333830058574677,0,2,18,17,1,3,-1.,18,18,1,1,3.,-6.8517157342284918e-004,-0.6008226275444031,0.0877470001578331,0,2,7,1,6,2,-1.,7,2,6,1,2.,5.3705149330198765e-003,-0.1231144964694977,0.3833355009555817,0,2,6,0,6,4,-1.,6,2,6,2,2.,-0.0134035395458341,0.3387736976146698,-0.1014048978686333,0,2,8,18,6,2,-1.,10,18,2,2,3.,-6.6856360062956810e-003,-0.6119359731674194,0.0477402210235596,0,2,7,6,5,2,-1.,7,7,5,1,2.,-4.2887418530881405e-003,0.2527579069137573,-0.1443451046943665,0,2,6,7,3,6,-1.,7,7,1,6,3.,-0.0108767496421933,0.5477573275566101,-0.0594554804265499,0,3,18,18,2,2,-1.,18,18,1,1,2.,19,19,1,1,2.,3.7882640026509762e-004,0.0834103003144264,-0.4422636926174164,0,2,16,8,3,7,-1.,17,8,1,7,3.,-2.4550149682909250e-003,0.2333099991083145,-0.1396448016166687,0,2,0,16,2,3,-1.,0,17,2,1,3.,1.2721839593723416e-003,0.0604802891612053,-0.4945608973503113,0,2,5,19,6,1,-1.,7,19,2,1,3.,-4.8933159559965134e-003,-0.6683326959609985,0.0462184995412827,0,2,9,5,6,6,-1.,9,7,6,2,3.,0.0264499895274639,-0.0732353627681732,0.4442596137523651,0,2,0,10,2,4,-1.,0,12,2,2,2.,-3.3706070389598608e-003,-0.4246433973312378,0.0686765611171722,0,2,0,9,4,3,-1.,2,9,2,3,2.,-2.9559480026364326e-003,0.1621803939342499,-0.1822299957275391,0,2,1,10,6,9,-1.,3,10,2,9,3.,0.0306199099868536,-0.0586433410644531,0.5326362848281860,0,2,9,0,6,2,-1.,11,0,2,2,3.,-9.5765907317399979e-003,-0.6056268215179443,0.0533459894359112,0,2,14,1,2,1,-1.,15,1,1,1,2.,6.6372493165545166e-005,-0.1668083965778351,0.1928416043519974,0,2,0,8,1,4,-1.,0,10,1,2,2.,5.0975950434803963e-003,0.0441195107996464,-0.5745884180068970,0,3,15,6,2,2,-1.,15,6,1,1,2.,16,7,1,1,2.,3.7112718564458191e-004,-0.1108639985322952,0.2310539036989212,0,2,7,5,3,6,-1.,8,5,1,6,3.,-8.6607588455080986e-003,0.4045628905296326,-0.0624460913240910,0,2,19,17,1,3,-1.,19,18,1,1,3.,8.7489158613607287e-004,0.0648751482367516,-0.4487104117870331,0,2,7,10,3,1,-1.,8,10,1,1,3.,1.1120870476588607e-003,-0.0938614606857300,0.3045391142368317,0,2,12,1,6,6,-1.,14,1,2,6,3.,-0.0238378196954727,-0.5888742804527283,0.0466594211757183,0,2,15,5,2,1,-1.,16,5,1,1,2.,2.2272899514064193e-004,-0.1489859968423843,0.1770195066928864,0,2,8,2,7,4,-1.,8,4,7,2,2.,0.0244674701243639,-0.0557896010577679,0.4920830130577087,0,2,4,0,14,15,-1.,4,5,14,5,3.,-0.1423932015895844,0.1519200056791306,-0.1877889931201935,0,2,7,8,6,6,-1.,9,8,2,6,3.,-0.0201231203973293,0.2178010046482086,-0.1208190023899078,0,2,11,17,1,3,-1.,11,18,1,1,3.,1.1513679783092812e-004,-0.1685658991336823,0.1645192950963974,0,3,12,16,2,4,-1.,12,16,1,2,2.,13,18,1,2,2.,-2.7556740678846836e-003,-0.6944203972816467,0.0394494682550430,0,2,10,13,2,1,-1.,11,13,1,1,2.,-7.5843912782147527e-005,0.1894136965274811,-0.1518384069204330,0,2,11,8,3,3,-1.,12,8,1,3,3.,-7.0697711780667305e-003,0.4706459939479828,-0.0579276196658611,0,2,2,0,6,8,-1.,4,0,2,8,3.,-0.0373931787908077,-0.7589244842529297,0.0341160483658314,0,3,3,5,6,6,-1.,3,5,3,3,2.,6,8,3,3,2.,-0.0159956105053425,0.3067046999931335,-0.0875255763530731,0,2,10,8,3,3,-1.,11,8,1,3,3.,-3.1183990649878979e-003,0.2619537115097046,-0.0912148877978325,0,2,5,17,4,2,-1.,5,18,4,1,2.,1.0651360498741269e-003,-0.1742756068706513,0.1527764052152634,0,2,8,16,5,2,-1.,8,17,5,1,2.,-1.6029420075938106e-003,0.3561263084411621,-0.0766299962997437,0,2,0,4,3,3,-1.,0,5,3,1,3.,4.3619908392429352e-003,0.0493569709360600,-0.5922877192497253,0,2,6,3,6,2,-1.,8,3,2,2,3.,-0.0107799097895622,-0.6392217874526978,0.0332045406103134,0,2,4,4,9,3,-1.,7,4,3,3,3.,-4.3590869754552841e-003,0.1610738933086395,-0.1522132009267807,0,2,0,13,1,4,-1.,0,15,1,2,2.,7.4596069753170013e-003,0.0331729613244534,-0.7500774264335632,0,2,0,17,8,3,-1.,0,18,8,1,3.,8.1385448575019836e-003,0.0263252798467875,-0.7173116207122803,0,2,6,1,11,6,-1.,6,3,11,2,3.,-0.0333384908735752,0.3353661000728607,-0.0708035901188850,-1.1418989896774292,55,0,2,4,10,6,2,-1.,6,10,2,2,3.,0.0195539798587561,-0.1043972000479698,0.5312895178794861,0,2,10,8,1,12,-1.,10,14,1,6,2.,0.0221229195594788,-0.2474727034568787,0.2084725052118301,0,2,5,8,3,4,-1.,6,8,1,4,3.,-4.1829389519989491e-003,0.3828943967819214,-0.1471157968044281,0,2,0,17,1,3,-1.,0,18,1,1,3.,-8.6381728760898113e-004,-0.6263288855552673,0.1199325993657112,0,2,0,17,1,3,-1.,0,18,1,1,3.,7.9958612332120538e-004,0.0925734713673592,-0.5516883134841919,0,2,13,8,3,4,-1.,14,8,1,4,3.,9.1527570039033890e-003,-0.0729298070073128,0.5551251173019409,0,2,1,5,5,4,-1.,1,7,5,2,2.,-3.9388681761920452e-003,0.2019603997468948,-0.2091203927993774,0,2,18,14,1,2,-1.,18,15,1,1,2.,1.4613410166930407e-004,-0.2786181867122650,0.1381741017103195,0,2,13,8,2,4,-1.,14,8,1,4,2.,-3.1691689509898424e-003,0.3668589890003204,-0.0763082429766655,0,2,10,6,6,8,-1.,12,6,2,8,3.,-0.0221893899142742,0.3909659981727600,-0.1097154021263123,0,2,8,6,6,10,-1.,10,6,2,10,3.,-7.4523608200252056e-003,0.1283859014511108,-0.2415986955165863,0,2,17,16,1,3,-1.,17,17,1,1,3.,7.7997002517804503e-004,0.0719780698418617,-0.4397650063037872,0,2,1,7,2,10,-1.,2,7,1,10,2.,-4.6783639118075371e-003,0.2156984955072403,-0.1420592069625855,0,2,5,9,6,3,-1.,7,9,2,3,3.,-0.0151886399835348,0.3645878136157990,-0.0826759263873100,0,2,0,8,5,12,-1.,0,14,5,6,2.,5.0619798712432384e-003,-0.3438040912151337,0.0920682325959206,0,2,0,11,1,3,-1.,0,12,1,1,3.,-1.7351920250803232e-003,-0.6172549724578857,0.0492144785821438,0,2,6,16,6,4,-1.,8,16,2,4,3.,-0.0124234501272440,-0.5855895280838013,0.0461126007139683,0,2,0,6,2,6,-1.,0,8,2,2,3.,-0.0130314296111465,-0.5971078872680664,0.0406724587082863,0,2,11,18,2,1,-1.,12,18,1,1,2.,-1.2369629694148898e-003,-0.6833416819572449,0.0331561788916588,0,2,5,1,9,2,-1.,5,2,9,1,2.,6.1022108420729637e-003,-0.0947292372584343,0.3010224103927612,0,2,0,0,1,2,-1.,0,1,1,1,2.,6.6952849738299847e-004,0.0818168669939041,-0.3519603013992310,0,2,15,9,3,3,-1.,16,9,1,3,3.,-1.7970580374822021e-003,0.2371897995471954,-0.1176870986819267,0,2,18,16,1,3,-1.,18,17,1,1,3.,-7.1074528386816382e-004,-0.4476378858089447,0.0576824806630611,0,2,11,10,6,1,-1.,13,10,2,1,3.,-5.9126471169292927e-003,0.4342541098594666,-0.0668685734272003,0,2,1,3,4,4,-1.,3,3,2,4,2.,-3.3132149837911129e-003,0.1815001070499420,-0.1418032050132752,0,2,11,2,1,18,-1.,11,8,1,6,3.,-0.0608146600425243,0.4722171127796173,-0.0614106394350529,0,2,9,1,5,12,-1.,9,5,5,4,3.,-0.0967141836881638,0.2768316864967346,-0.0944900363683701,0,2,12,0,8,1,-1.,16,0,4,1,2.,3.9073550142347813e-003,-0.1227853000164032,0.2105740010738373,0,2,8,6,3,10,-1.,9,6,1,10,3.,-9.0431869029998779e-003,0.3564156889915466,-0.0778062269091606,0,2,19,2,1,6,-1.,19,4,1,2,3.,-4.8800031654536724e-003,-0.4103479087352753,0.0696943774819374,0,2,18,6,2,2,-1.,18,7,2,1,2.,-4.3547428213059902e-003,-0.7301788926124573,0.0366551503539085,0,2,7,7,3,4,-1.,8,7,1,4,3.,-9.6500627696514130e-003,0.5518112778663635,-0.0531680807471275,0,2,5,0,6,5,-1.,7,0,2,5,3.,-0.0173973105847836,-0.5708423256874085,0.0502140894532204,0,2,0,3,7,3,-1.,0,4,7,1,3.,-6.8304329179227352e-003,-0.4618028104305267,0.0502026900649071,0,2,1,6,2,1,-1.,2,6,1,1,2.,3.3255619928240776e-004,-0.0953627303242683,0.2598375976085663,0,3,4,8,2,10,-1.,4,8,1,5,2.,5,13,1,5,2.,-2.3100529797375202e-003,0.2287247031927109,-0.1053353026509285,0,3,2,18,18,2,-1.,2,18,9,1,2.,11,19,9,1,2.,-7.5426651164889336e-003,-0.5699051022529602,0.0488634593784809,0,3,2,7,4,4,-1.,2,7,2,2,2.,4,9,2,2,2.,-5.2723060362040997e-003,0.3514518141746521,-0.0823901072144508,0,2,17,3,3,4,-1.,18,3,1,4,3.,-4.8578968271613121e-003,-0.6041762232780457,0.0445394404232502,0,3,16,9,2,8,-1.,16,9,1,4,2.,17,13,1,4,2.,1.5867310576140881e-003,-0.1034090965986252,0.2328201979398727,0,2,15,7,1,6,-1.,15,9,1,2,3.,-4.7427811659872532e-003,0.2849028110504150,-0.0980904996395111,0,2,14,2,2,2,-1.,14,3,2,1,2.,-1.3515240279957652e-003,0.2309643030166626,-0.1136184036731720,0,2,17,0,2,3,-1.,17,1,2,1,3.,2.2526069078594446e-003,0.0644783228635788,-0.4220589101314545,0,3,16,18,2,2,-1.,16,18,1,1,2.,17,19,1,1,2.,-3.8038659840822220e-004,-0.3807620108127594,0.0600432902574539,0,2,10,4,4,3,-1.,10,5,4,1,3.,4.9043921753764153e-003,-0.0761049985885620,0.3323217034339905,0,2,0,2,8,6,-1.,4,2,4,6,2.,-9.0969670563936234e-003,0.1428779065608978,-0.1688780039548874,0,2,7,14,6,6,-1.,7,16,6,2,3.,-6.9317929446697235e-003,0.2725540995597839,-0.0928795635700226,0,2,11,15,2,2,-1.,11,16,2,1,2.,1.1471060570329428e-003,-0.1527305990457535,0.1970240026712418,0,2,7,1,9,4,-1.,10,1,3,4,3.,-0.0376628898084164,-0.5932043790817261,0.0407386012375355,0,2,9,7,3,7,-1.,10,7,1,7,3.,-6.8165571428835392e-003,0.2549408972263336,-0.0940819606184959,0,3,6,17,2,2,-1.,6,17,1,1,2.,7,18,1,1,2.,6.6205562325194478e-004,0.0467957183718681,-0.4845437109470367,0,2,4,6,3,9,-1.,5,6,1,9,3.,-4.2202551849186420e-003,0.2468214929103851,-0.0946739763021469,0,2,0,10,19,10,-1.,0,15,19,5,2.,-0.0689865127205849,-0.6651480197906494,0.0359263904392719,0,2,5,17,6,1,-1.,7,17,2,1,3.,6.1707608401775360e-003,0.0258333198726177,-0.7268627285957336,0,2,0,12,6,3,-1.,3,12,3,3,2.,0.0105362497270107,-0.0818289965391159,0.2976079881191254,-1.1255199909210205,32,0,2,2,5,18,5,-1.,8,5,6,5,3.,-0.0627587288618088,0.2789908051490784,-0.2965610921382904,0,2,1,15,6,4,-1.,1,17,6,2,2.,3.4516479354351759e-003,-0.3463588058948517,0.2090384066104889,0,2,14,10,6,6,-1.,16,10,2,6,3.,-7.8699486330151558e-003,0.2414488941431046,-0.1920557022094727,0,2,0,14,4,3,-1.,0,15,4,1,3.,-3.4624869003891945e-003,-0.5915178060531616,0.1248644962906838,0,2,1,7,6,11,-1.,3,7,2,11,3.,-9.4818761572241783e-003,0.1839154064655304,-0.2485826015472412,0,2,13,17,7,2,-1.,13,18,7,1,2.,2.3226840130519122e-004,-0.3304725885391235,0.1099926009774208,0,2,0,14,2,3,-1.,0,15,2,1,3.,1.8101120367646217e-003,0.0987440124154091,-0.4963478147983551,0,2,0,0,6,2,-1.,3,0,3,2,2.,-5.4422430694103241e-003,0.2934441864490509,-0.1309475004673004,0,2,0,1,6,3,-1.,3,1,3,3,2.,7.4148122221231461e-003,-0.1476269960403442,0.3327716886997223,0,2,0,8,2,6,-1.,0,10,2,2,3.,-0.0155651401728392,-0.6840490102767944,0.0998726934194565,0,3,1,2,6,14,-1.,1,2,3,7,2.,4,9,3,7,2.,0.0287205204367638,-0.1483328044414520,0.3090257942676544,0,3,17,5,2,2,-1.,17,5,1,1,2.,18,6,1,1,2.,9.6687392215244472e-005,-0.1743104010820389,0.2140295952558518,0,2,11,10,9,4,-1.,14,10,3,4,3.,0.0523710586130619,-0.0701568573713303,0.4922299087047577,0,2,2,9,12,4,-1.,6,9,4,4,3.,-0.0864856913685799,0.5075724720954895,-0.0752942115068436,0,2,7,10,12,2,-1.,11,10,4,2,3.,-0.0421698689460754,0.4568096101284027,-0.0902199000120163,0,2,2,13,1,2,-1.,2,14,1,1,2.,4.5369830331765115e-005,-0.2653827965259552,0.1618953943252564,0,2,16,7,4,3,-1.,16,8,4,1,3.,5.2918000146746635e-003,0.0748901516199112,-0.5405467152595520,0,2,19,16,1,3,-1.,19,17,1,1,3.,-7.5511651812121272e-004,-0.4926199018955231,0.0587239488959312,0,2,18,11,1,2,-1.,18,12,1,1,2.,7.5108138844370842e-005,-0.2143210023641586,0.1407776027917862,0,3,12,7,8,2,-1.,12,7,4,1,2.,16,8,4,1,2.,4.9981209449470043e-003,-0.0905473381280899,0.3571606874465942,0,2,14,9,2,4,-1.,15,9,1,4,2.,-1.4929979806765914e-003,0.2562345862388611,-0.1422906965017319,0,3,14,2,6,4,-1.,14,2,3,2,2.,17,4,3,2,2.,2.7239411137998104e-003,-0.1564925014972687,0.2108871042728424,0,2,14,0,6,1,-1.,17,0,3,1,2.,2.2218320518732071e-003,-0.1507298946380615,0.2680186927318573,0,2,3,12,2,1,-1.,4,12,1,1,2.,-7.3993072146549821e-004,0.2954699099063873,-0.1069239005446434,0,2,17,2,3,1,-1.,18,2,1,1,3.,2.0113459322601557e-003,0.0506143495440483,-0.7168337106704712,0,2,1,16,18,2,-1.,7,16,6,2,3.,0.0114528704434633,-0.1271906942129135,0.2415277957916260,0,2,2,19,8,1,-1.,6,19,4,1,2.,-1.0782170575112104e-003,0.2481300979852676,-0.1346119940280914,0,2,1,17,4,3,-1.,1,18,4,1,3.,3.3417691010981798e-003,0.0535783097147942,-0.5227416753768921,0,2,19,13,1,2,-1.,19,14,1,1,2.,6.9398651248775423e-005,-0.2169874012470245,0.1281217932701111,0,3,9,16,10,4,-1.,9,16,5,2,2.,14,18,5,2,2.,-4.0982551872730255e-003,0.2440188974142075,-0.1157058998942375,0,3,12,9,2,4,-1.,12,9,1,2,2.,13,11,1,2,2.,-1.6289720078930259e-003,0.2826147079467773,-0.1065946966409683,0,2,19,11,1,9,-1.,19,14,1,3,3.,0.0139848599210382,0.0427158996462822,-0.7364631295204163,-1.1729990243911743,30,0,2,6,6,14,14,-1.,6,13,14,7,2.,0.1641651988029480,-0.4896030128002167,0.1760770976543427,0,2,2,17,4,2,-1.,2,18,4,1,2.,8.3413062384352088e-004,-0.2822043001651764,0.2419957965612412,0,2,0,2,1,3,-1.,0,3,1,1,3.,-1.7193210078403354e-003,-0.7148588895797730,0.0861622169613838,0,2,0,12,1,3,-1.,0,13,1,1,3.,-1.5654950402677059e-003,-0.7297238111495972,0.0940706729888916,0,2,15,15,4,4,-1.,15,17,4,2,2.,1.9124479731544852e-003,-0.3118715882301331,0.1814339011907578,0,2,2,5,18,7,-1.,8,5,6,7,3.,-0.1351236999034882,0.2957729995250702,-0.2217925041913986,0,2,1,16,5,3,-1.,1,17,5,1,3.,-4.0300549007952213e-003,-0.6659513711929321,0.0854310169816017,0,2,0,4,2,3,-1.,0,5,2,1,3.,-2.8640460222959518e-003,-0.6208636164665222,0.0531060211360455,0,2,0,6,2,6,-1.,1,6,1,6,2.,-1.4065420255064964e-003,0.2234628945589066,-0.2021100968122482,0,2,16,14,4,3,-1.,16,15,4,1,3.,-3.5820449702441692e-003,-0.5403040051460266,0.0682136192917824,0,3,0,0,10,6,-1.,0,0,5,3,2.,5,3,5,3,2.,0.0415444709360600,-0.0652158409357071,0.6210923194885254,0,2,2,2,3,6,-1.,3,2,1,6,3.,-9.1709550470113754e-003,-0.7555329799652100,0.0526404492557049,0,2,2,0,3,10,-1.,3,0,1,10,3.,6.1552738770842552e-003,0.0909394025802612,-0.4424613118171692,0,2,5,5,2,2,-1.,5,6,2,1,2.,-1.0043520014733076e-003,0.2429233044385910,-0.1866979002952576,0,2,12,6,4,4,-1.,12,8,4,2,2.,0.0115198297426105,-0.1176315024495125,0.3672345876693726,0,2,13,5,7,3,-1.,13,6,7,1,3.,-8.9040733873844147e-003,-0.4893133044242859,0.1089702025055885,0,2,10,13,1,2,-1.,10,14,1,1,2.,5.3973670583218336e-004,-0.2185039967298508,0.1848998963832855,0,2,16,16,4,2,-1.,18,16,2,2,2.,1.3727260520681739e-003,-0.1507291048765183,0.2917312979698181,0,2,16,12,4,7,-1.,18,12,2,7,2.,-0.0108073903247714,0.4289745092391968,-0.1028013974428177,0,2,16,17,1,3,-1.,16,18,1,1,3.,1.2670770520344377e-003,0.0741921588778496,-0.6420825123786926,0,2,19,9,1,3,-1.,19,10,1,1,3.,2.2991129662841558e-003,0.0471002794802189,-0.7233523130416870,0,2,18,7,2,6,-1.,19,7,1,6,2.,2.7187510859221220e-003,-0.1708686947822571,0.2351350933313370,0,2,8,1,3,4,-1.,9,1,1,4,3.,-6.6619180142879486e-003,-0.7897542715072632,0.0450846701860428,0,2,14,0,6,9,-1.,16,0,2,9,3.,-0.0482666492462158,-0.6957991719245911,0.0419760793447495,0,2,4,2,10,2,-1.,9,2,5,2,2.,0.0152146900072694,-0.1081828027963638,0.3646062016487122,0,3,2,12,8,4,-1.,2,12,4,2,2.,6,14,4,2,2.,-6.0080131515860558e-003,0.3097099065780640,-0.1135921031236649,0,2,0,4,7,3,-1.,0,5,7,1,3.,6.6127157770097256e-003,0.0806653425097466,-0.4665853083133698,0,2,14,14,3,3,-1.,15,14,1,3,3.,-7.9607013612985611e-003,-0.8720194101333618,0.0367745906114578,0,2,0,3,4,3,-1.,2,3,2,3,2.,3.8847199175506830e-003,-0.1166628971695900,0.3307026922702789,0,2,1,0,2,7,-1.,2,0,1,7,2.,-1.0988810099661350e-003,0.2387257069349289,-0.1765675991773605,-1.0368299484252930,44,0,2,15,16,4,4,-1.,15,18,4,2,2.,3.5903379321098328e-003,-0.2368807941675186,0.2463164031505585,0,2,5,8,12,4,-1.,5,10,12,2,2.,6.4815930090844631e-003,-0.3137362003326416,0.1867575943470001,0,2,3,17,1,2,-1.,3,18,1,1,2.,7.3048402555286884e-005,-0.2764435112476349,0.1649623960256577,0,2,6,1,3,4,-1.,7,1,1,4,3.,-3.8514640182256699e-003,-0.5601450800895691,0.1129473969340324,0,2,6,2,3,4,-1.,7,2,1,4,3.,3.8588210009038448e-003,0.0398489981889725,-0.5807185769081116,0,2,6,8,9,12,-1.,9,8,3,12,3.,-0.0246512200683355,0.1675501018762589,-0.2534367144107819,0,2,8,1,8,6,-1.,8,3,8,2,3.,0.0472455210983753,-0.1066208034753799,0.3945198059082031,0,2,14,2,6,3,-1.,17,2,3,3,2.,6.5964651294052601e-003,-0.1774425059556961,0.2728019058704376,0,2,0,6,1,3,-1.,0,7,1,1,3.,-1.3177490327507257e-003,-0.5427265167236328,0.0486065894365311,0,2,10,0,10,2,-1.,15,0,5,2,2.,-5.0261709839105606e-003,0.2439424991607666,-0.1314364969730377,0,2,11,0,3,2,-1.,12,0,1,2,3.,3.4632768947631121e-003,0.0690493434667587,-0.7033624053001404,0,2,3,19,10,1,-1.,8,19,5,1,2.,2.1692588925361633e-003,-0.1328946053981781,0.2209852933883667,0,2,0,4,7,16,-1.,0,12,7,8,2.,0.0293958708643913,-0.2853052020072937,0.1354399025440216,0,2,2,16,1,3,-1.,2,17,1,1,3.,-9.6181448316201568e-004,-0.5804138183593750,0.0374506488442421,0,2,7,8,12,6,-1.,11,8,4,6,3.,-0.1082099974155426,0.3946728110313416,-0.0786559432744980,0,2,14,9,6,7,-1.,16,9,2,7,3.,-0.0180248692631722,0.2735562920570374,-0.1341529935598373,0,2,12,17,6,1,-1.,14,17,2,1,3.,6.2509840354323387e-003,0.0233880598098040,-0.8008859157562256,0,2,16,1,3,1,-1.,17,1,1,1,3.,-1.6088379779830575e-003,-0.5676252245903015,0.0412156693637371,0,3,0,17,8,2,-1.,0,17,4,1,2.,4,18,4,1,2.,7.7564752427861094e-004,-0.1489126980304718,0.1908618062734604,0,2,17,0,2,1,-1.,18,0,1,1,2.,8.7122338300105184e-005,-0.1555753052234650,0.1942822039127350,0,2,4,15,6,5,-1.,6,15,2,5,3.,-0.0207553207874298,-0.6300653219223023,0.0361343808472157,0,2,7,2,8,2,-1.,7,3,8,1,2.,-6.2931738793849945e-003,0.2560924887657166,-0.1058826968073845,0,2,4,1,8,4,-1.,4,3,8,2,2.,0.0108441496267915,-0.1012485027313232,0.3032212853431702,0,2,5,19,2,1,-1.,6,19,1,1,2.,-6.3752777350600809e-005,0.1911157965660095,-0.1384923011064529,0,2,5,19,2,1,-1.,6,19,1,1,2.,6.6480963141657412e-005,-0.1520525068044663,0.2170630991458893,0,2,16,17,1,3,-1.,16,18,1,1,3.,1.3560829684138298e-003,0.0494317896664143,-0.6427984237670898,0,2,0,11,2,3,-1.,1,11,1,3,2.,-9.0662558795884252e-004,0.1798201054334641,-0.1404460966587067,0,2,0,19,4,1,-1.,2,19,2,1,2.,1.0473709553480148e-003,-0.1093354970216751,0.2426594048738480,0,2,0,18,4,2,-1.,2,18,2,2,2.,-1.0243969736620784e-003,0.2716268002986908,-0.1182091981172562,0,2,2,17,1,3,-1.,2,18,1,1,3.,-1.2024149764329195e-003,-0.7015110254287720,0.0394898988306522,0,2,5,7,11,2,-1.,5,8,11,1,2.,7.6911649666726589e-003,-0.0922189131379128,0.3104628920555115,0,2,9,2,4,10,-1.,9,7,4,5,2.,-0.1396654993295670,0.6897938847541809,-0.0397061184048653,0,2,0,2,4,3,-1.,0,3,4,1,3.,2.1276050247251987e-003,0.0972776114940643,-0.2884179949760437,0,2,10,19,10,1,-1.,15,19,5,1,2.,-2.7594310231506824e-003,0.2416867017745972,-0.1127782016992569,0,2,11,17,8,3,-1.,15,17,4,3,2.,5.2236132323741913e-003,-0.1143027991056442,0.2425678074359894,0,2,8,19,3,1,-1.,9,19,1,1,3.,-1.2590440455824137e-003,-0.5967938899993897,0.0476639606058598,0,2,14,0,3,4,-1.,15,0,1,4,3.,-3.7192099262028933e-003,-0.4641413092613220,0.0528476908802986,0,2,10,6,4,3,-1.,10,7,4,1,3.,5.9696151874959469e-003,-0.0732442885637283,0.3874309062957764,0,2,0,8,3,2,-1.,0,9,3,1,2.,-5.1776720210909843e-003,-0.7419322729110718,0.0404967106878757,0,2,7,12,3,6,-1.,7,14,3,2,3.,5.0035100430250168e-003,-0.1388880014419556,0.1876762062311173,0,2,1,18,1,2,-1.,1,19,1,1,2.,-5.2013457752764225e-004,-0.5494061708450317,0.0494178496301174,0,2,0,12,4,4,-1.,2,12,2,4,2.,5.3168768063187599e-003,-0.0824829787015915,0.3174056112766266,0,2,1,8,6,7,-1.,3,8,2,7,3.,-0.0147745897993445,0.2081609964370728,-0.1211555972695351,0,2,0,8,4,5,-1.,2,8,2,5,2.,-0.0414164513349533,-0.8243780732154846,0.0333291888237000,-1.0492420196533203,53,0,2,19,16,1,3,-1.,19,17,1,1,3.,9.0962520334869623e-004,0.0845799669623375,-0.5611841082572937,0,2,1,5,18,6,-1.,7,5,6,6,3.,-0.0561397895216942,0.1534174978733063,-0.2696731984615326,0,2,2,15,4,2,-1.,2,16,4,1,2.,1.0292009683325887e-003,-0.2048998028039932,0.2015317976474762,0,2,18,6,2,11,-1.,19,6,1,11,2.,2.8783010784536600e-003,-0.1735114008188248,0.2129794955253601,0,2,0,12,2,6,-1.,0,14,2,2,3.,-7.4144392274320126e-003,-0.5962486863136292,0.0470779500901699,0,2,12,5,3,2,-1.,12,6,3,1,2.,-1.4831849839538336e-003,0.1902461051940918,-0.1598639041185379,0,2,1,3,2,3,-1.,1,4,2,1,3.,4.5968941412866116e-003,0.0314471311867237,-0.6869434118270874,0,2,16,14,4,4,-1.,16,16,4,2,2.,2.4255330208688974e-003,-0.2360935956239700,0.1103610992431641,0,2,6,8,12,5,-1.,10,8,4,5,3.,-0.0849505662918091,0.2310716062784195,-0.1377653032541275,0,2,13,7,2,7,-1.,14,7,1,7,2.,-5.0145681016147137e-003,0.3867610991001129,-0.0562173798680305,0,2,1,8,2,6,-1.,2,8,1,6,2.,-2.1482061129063368e-003,0.1819159984588623,-0.1761569976806641,0,2,15,0,3,7,-1.,16,0,1,7,3.,-0.0103967702016234,-0.7535138130187988,0.0240919701755047,0,2,4,2,6,2,-1.,6,2,2,2,3.,-0.0134667502716184,-0.7211886048316956,0.0349493697285652,0,2,0,9,20,9,-1.,0,12,20,3,3.,-0.0844354778528214,-0.3379263877868652,0.0711138173937798,0,2,10,14,2,2,-1.,10,15,2,1,2.,2.4771490134298801e-003,-0.1176510974764824,0.2254198938608170,0,2,6,5,10,4,-1.,6,7,10,2,2.,0.0158280506730080,-0.0695362165570259,0.3139536976814270,0,2,6,1,5,9,-1.,6,4,5,3,3.,0.0649169832468033,-0.0750435888767242,0.4067733883857727,0,3,16,18,2,2,-1.,16,18,1,1,2.,17,19,1,1,2.,2.9652469675056636e-004,0.0739533603191376,-0.3454400897026062,0,2,0,14,2,4,-1.,0,16,2,2,2.,1.3129520229995251e-003,-0.1690943986177445,0.1525837033987045,0,2,10,8,2,5,-1.,11,8,1,5,2.,-5.8032129891216755e-003,0.3526014983654022,-0.0834440663456917,0,2,3,7,12,7,-1.,7,7,4,7,3.,-0.1479167938232422,0.4300465881824493,-0.0573099292814732,0,2,0,0,6,6,-1.,3,0,3,6,2.,-0.0165841504931450,0.2343268990516663,-0.1090764030814171,0,2,1,0,4,4,-1.,3,0,2,4,2.,3.0183270573616028e-003,-0.1360093951225281,0.2640928924083710,0,2,0,0,6,8,-1.,2,0,2,8,3.,-0.0364719182252884,-0.6280974149703980,0.0435451082885265,0,2,0,0,2,1,-1.,1,0,1,1,2.,-7.3119226726703346e-005,0.1647063046693802,-0.1646378040313721,0,2,0,0,3,3,-1.,0,1,3,1,3.,-3.6719450727105141e-003,-0.4742136001586914,0.0485869199037552,0,2,5,4,2,4,-1.,5,6,2,2,2.,-4.0151178836822510e-003,0.1822218000888825,-0.1409751027822495,0,2,2,10,9,1,-1.,5,10,3,1,3.,0.0199480205774307,-0.0697876587510109,0.3670746088027954,0,2,1,17,1,3,-1.,1,18,1,1,3.,7.6699437340721488e-004,0.0557292997837067,-0.4458543062210083,0,2,0,17,2,3,-1.,0,18,2,1,3.,-1.1806039838120341e-003,-0.4687662124633789,0.0489022210240364,0,2,0,15,16,3,-1.,8,15,8,3,2.,0.0158473495393991,-0.1212020963430405,0.2056653052568436,0,2,0,5,4,1,-1.,2,5,2,1,2.,-1.1985700111836195e-003,0.2026209980249405,-0.1282382011413574,0,2,1,0,6,20,-1.,3,0,2,20,3.,-0.1096495985984802,-0.8661919236183167,0.0303518492728472,0,3,2,5,4,6,-1.,2,5,2,3,2.,4,8,2,3,2.,-9.2532606795430183e-003,0.2934311926364899,-0.0853619500994682,0,2,9,16,6,3,-1.,11,16,2,3,3.,0.0146865304559469,0.0327986218035221,-0.7755656242370606,0,2,11,17,6,1,-1.,14,17,3,1,2.,-1.3514430029317737e-003,0.2442699968814850,-0.1150325015187264,0,2,3,17,15,2,-1.,8,17,5,2,3.,-4.3728090822696686e-003,0.2168767005205154,-0.1398448050022125,0,2,18,0,2,3,-1.,18,1,2,1,3.,3.4263390116393566e-003,0.0456142202019691,-0.5456771254539490,0,2,13,1,7,4,-1.,13,3,7,2,2.,-3.8404068909585476e-003,0.1494950056076050,-0.1506250947713852,0,3,13,6,4,4,-1.,13,6,2,2,2.,15,8,2,2,2.,3.7988980766385794e-003,-0.0873016268014908,0.2548153102397919,0,2,17,6,3,4,-1.,17,8,3,2,2.,-2.0094281062483788e-003,0.1725907027721405,-0.1428847014904022,0,2,14,9,2,2,-1.,15,9,1,2,2.,-2.4370709434151649e-003,0.2684809863567352,-0.0818982198834419,0,2,17,17,1,3,-1.,17,18,1,1,3.,1.0485399980098009e-003,0.0461132600903511,-0.4724327921867371,0,2,3,19,8,1,-1.,7,19,4,1,2.,1.7460780218243599e-003,-0.1103043034672737,0.2037972956895828,0,2,0,9,3,6,-1.,0,12,3,3,2.,5.8608627878129482e-003,-0.1561965942382813,0.1592743992805481,0,2,4,7,15,5,-1.,9,7,5,5,3.,-0.0277249794453382,0.1134911999106407,-0.2188514024019241,0,2,6,9,9,5,-1.,9,9,3,5,3.,0.0470806397497654,-0.0416887290775776,0.5363004803657532,0,2,8,1,6,2,-1.,10,1,2,2,3.,-7.9283770173788071e-003,-0.5359513163566589,0.0442375093698502,0,2,4,0,12,2,-1.,10,0,6,2,2.,-0.0128805404528975,0.2323794960975647,-0.1024625003337860,0,2,7,0,10,3,-1.,12,0,5,3,2.,0.0236047692596912,-0.0882914364337921,0.3056105971336365,0,2,5,0,9,6,-1.,5,2,9,2,3.,0.0159022007137537,-0.1223810985684395,0.1784912049770355,0,2,8,3,6,4,-1.,8,5,6,2,2.,7.9939495772123337e-003,-0.0837290063500404,0.3231959044933319,0,2,17,4,2,3,-1.,17,5,2,1,3.,5.7100867852568626e-003,0.0384792089462280,-0.6813815236091614,-1.1122100353240967,51,0,2,5,2,4,3,-1.,5,3,4,1,3.,2.2480720654129982e-003,-0.1641687005758286,0.4164853096008301,0,2,5,9,2,6,-1.,6,9,1,6,2.,4.5813550241291523e-003,-0.1246595978736877,0.4038512110710144,0,2,14,10,2,6,-1.,15,10,1,6,2.,-1.6073239967226982e-003,0.2608245909214020,-0.2028252035379410,0,2,7,4,3,3,-1.,7,5,3,1,3.,2.5205370038747787e-003,-0.1055722981691361,0.3666911125183106,0,3,12,4,8,2,-1.,12,4,4,1,2.,16,5,4,1,2.,2.4119189474731684e-003,-0.1387760043144226,0.2995991110801697,0,2,15,8,1,6,-1.,15,10,1,2,3.,5.7156179100275040e-003,-0.0776834636926651,0.4848192036151886,0,2,4,17,11,3,-1.,4,18,11,1,3.,3.1093840952962637e-003,-0.1122900024056435,0.2921550869941711,0,2,3,0,16,20,-1.,3,10,16,10,2.,-0.0868366286158562,-0.3677960038185120,0.0725972428917885,0,2,12,4,4,6,-1.,12,6,4,2,3.,5.2652182057499886e-003,-0.1089029014110565,0.3179126083850861,0,2,11,0,6,6,-1.,13,0,2,6,3.,-0.0199135299772024,-0.5337343811988831,0.0705857127904892,0,3,13,1,6,4,-1.,13,1,3,2,2.,16,3,3,2,2.,3.8297839928418398e-003,-0.1357591003179550,0.2278887927532196,0,2,11,0,6,4,-1.,13,0,2,4,3.,0.0104318596422672,0.0887979120016098,-0.4795897006988525,0,2,8,6,6,9,-1.,10,6,2,9,3.,-0.0200404394418001,0.1574553996324539,-0.1777157038450241,0,2,7,0,3,4,-1.,8,0,1,4,3.,-5.2967290394008160e-003,-0.6843491792678833,0.0356714613735676,0,3,0,17,14,2,-1.,0,17,7,1,2.,7,18,7,1,2.,-2.1624139044433832e-003,0.2831803858280182,-0.0985112786293030,0,3,6,18,2,2,-1.,6,18,1,1,2.,7,19,1,1,2.,-3.5464888787828386e-004,-0.3707734048366547,0.0809329524636269,0,2,18,17,1,3,-1.,18,18,1,1,3.,-1.8152060511056334e-004,-0.3220703005790710,0.0775510594248772,0,3,17,18,2,2,-1.,17,18,1,1,2.,18,19,1,1,2.,-2.7563021285459399e-004,-0.3244127929210663,0.0879494771361351,0,2,5,7,1,9,-1.,5,10,1,3,3.,6.3823810778558254e-003,-0.0889247134327888,0.3172721862792969,0,2,5,3,6,4,-1.,7,3,2,4,3.,0.0111509095877409,0.0710198432207108,-0.4049403965473175,0,3,1,9,6,2,-1.,1,9,3,1,2.,4,10,3,1,2.,-1.0593760525807738e-003,0.2605066895484924,-0.1176564022898674,0,2,6,9,2,3,-1.,7,9,1,3,2.,2.3906480055302382e-003,-0.0843886211514473,0.3123055100440979,0,2,6,8,6,12,-1.,8,8,2,12,3.,-0.0110007496550679,0.1915224939584732,-0.1521002054214478,0,3,4,18,2,2,-1.,4,18,1,1,2.,5,19,1,1,2.,-2.4643228971399367e-004,-0.3176515996456146,0.0865822583436966,0,2,9,1,6,6,-1.,9,3,6,2,3.,0.0230532698333263,-0.1008976027369499,0.2576929032802582,0,2,6,17,6,2,-1.,6,18,6,1,2.,-2.2135660983622074e-003,0.4568921029567719,-0.0524047911167145,0,2,3,18,16,2,-1.,3,19,16,1,2.,-9.7139709396287799e-004,-0.3551838099956513,0.0800943821668625,0,2,3,0,3,11,-1.,4,0,1,11,3.,1.5676229959353805e-003,0.1009142026305199,-0.2160304039716721,0,2,13,18,3,1,-1.,14,18,1,1,3.,7.5460801599547267e-004,0.0578961782157421,-0.4046111106872559,0,2,6,0,9,6,-1.,6,2,9,2,3.,-0.0206989701837301,0.3154363036155701,-0.0807130485773087,0,3,1,2,12,4,-1.,1,2,6,2,2.,7,4,6,2,2.,-0.0206199400126934,0.2718166112899780,-0.0763586163520813,0,2,3,3,6,4,-1.,5,3,2,4,3.,0.0216111298650503,0.0394934490323067,-0.5942965149879456,0,2,12,0,8,1,-1.,16,0,4,1,2.,6.5676742233335972e-003,-0.0983536690473557,0.2364927977323532,0,2,9,0,6,2,-1.,11,0,2,2,3.,-8.8434796780347824e-003,-0.5252342820167542,0.0430999211966991,0,2,3,3,12,1,-1.,9,3,6,1,2.,-9.4260741025209427e-003,0.2466513067483902,-0.0941307172179222,0,3,2,7,6,2,-1.,2,7,3,1,2.,5,8,3,1,2.,-1.9830230157822371e-003,0.2674370110034943,-0.0900693163275719,0,2,0,8,4,6,-1.,0,10,4,2,3.,-1.7358399927616119e-003,0.1594001948833466,-0.1578941047191620,0,2,9,6,3,7,-1.,10,6,1,7,3.,-0.0135138696059585,0.4079233109951019,-0.0642231181263924,0,2,9,6,6,13,-1.,11,6,2,13,3.,-0.0193940103054047,0.1801564991474152,-0.1373140066862106,0,2,11,12,6,1,-1.,13,12,2,1,3.,-3.2684770412743092e-003,0.2908039093017578,-0.0801619067788124,0,2,18,9,2,6,-1.,18,12,2,3,2.,4.1773589327931404e-004,-0.2141298055648804,0.1127343997359276,0,2,17,2,3,9,-1.,18,2,1,9,3.,-7.6351119205355644e-003,-0.4536595940589905,0.0546250604093075,0,3,13,8,4,6,-1.,13,8,2,3,2.,15,11,2,3,2.,-8.3652976900339127e-003,0.2647292017936707,-0.0943341106176376,0,2,4,2,12,6,-1.,10,2,6,6,2.,0.0277684498578310,-0.1013671010732651,0.2074397951364517,0,2,4,14,16,6,-1.,12,14,8,6,2.,-0.0548912286758423,0.2884030938148499,-0.0753120407462120,0,2,6,19,10,1,-1.,11,19,5,1,2.,2.5793339591473341e-003,-0.1108852997422218,0.2172496020793915,0,2,6,17,1,3,-1.,6,18,1,1,3.,6.6196516854688525e-005,-0.1887210011482239,0.1444068998098373,0,2,4,14,10,3,-1.,4,15,10,1,3.,5.0907251425087452e-003,-0.0776012316346169,0.2939837872982025,0,2,6,0,12,12,-1.,6,4,12,4,3.,-0.1044425964355469,0.2013310939073563,-0.1090397015213966,0,3,5,7,4,2,-1.,5,7,2,1,2.,7,8,2,1,2.,-6.7273090826347470e-004,0.1794590055942535,-0.1202367022633553,0,2,17,5,3,2,-1.,18,5,1,2,3.,3.2412849832326174e-003,0.0406881310045719,-0.5460057258605957,-1.2529590129852295,44,0,2,8,13,6,3,-1.,8,14,6,1,3.,5.2965320646762848e-003,-0.1215452998876572,0.6442037224769592,0,2,8,13,5,3,-1.,8,14,5,1,3.,-2.5326260365545750e-003,0.5123322010040283,-0.1110825985670090,0,2,13,2,1,18,-1.,13,11,1,9,2.,-2.9183230362832546e-003,-0.5061542987823486,0.1150197982788086,0,2,6,10,9,2,-1.,9,10,3,2,3.,-0.0236923396587372,0.3716728091239929,-0.1467268019914627,0,2,11,0,7,4,-1.,11,2,7,2,2.,0.0201774705201387,-0.1738884001970291,0.4775949120521545,0,2,1,0,6,8,-1.,3,0,2,8,3.,-0.0217232108116150,-0.4388009011745453,0.1357689946889877,0,2,9,15,3,3,-1.,9,16,3,1,3.,2.8369780629873276e-003,-0.1251206994056702,0.4678902924060822,0,2,9,17,9,3,-1.,9,18,9,1,3.,2.7148420922458172e-003,-0.0880188569426537,0.3686651885509491,0,2,12,12,3,3,-1.,12,13,3,1,3.,3.2625689636915922e-003,-0.0853353068232536,0.5164473056793213,0,2,4,1,3,5,-1.,5,1,1,5,3.,-3.5618850961327553e-003,-0.4450393021106720,0.0917381718754768,0,2,10,14,2,3,-1.,10,15,2,1,3.,1.9227749435231090e-003,-0.1107731014490128,0.3941699862480164,0,3,18,17,2,2,-1.,18,17,1,1,2.,19,18,1,1,2.,-3.5111969918943942e-004,-0.3777570128440857,0.1216617003083229,0,3,18,18,2,2,-1.,18,18,1,1,2.,19,19,1,1,2.,1.9121779769193381e-004,0.0748160183429718,-0.4076710045337677,0,3,18,18,2,2,-1.,18,18,1,1,2.,19,19,1,1,2.,-2.6525629800744355e-004,-0.3315171897411346,0.1129112020134926,0,2,4,10,9,1,-1.,7,10,3,1,3.,0.0200867000967264,-0.0615981183946133,0.5612881779670715,0,2,3,9,6,5,-1.,5,9,2,5,3.,0.0367832481861115,-0.0602513886988163,0.5219249129295349,0,2,18,8,1,12,-1.,18,14,1,6,2.,1.3941619545221329e-003,-0.3550305068492889,0.1086302027106285,0,3,0,2,8,6,-1.,0,2,4,3,2.,4,5,4,3,2.,-0.0151816699653864,0.2273965030908585,-0.1625299006700516,0,2,9,4,3,3,-1.,9,5,3,1,3.,4.6796840615570545e-003,-0.0575350411236286,0.4812423884868622,0,3,3,18,2,2,-1.,3,18,1,1,2.,4,19,1,1,2.,-1.7988319450523704e-004,-0.3058767020702362,0.1086815968155861,0,2,6,4,4,3,-1.,6,5,4,1,3.,-3.5850999411195517e-003,0.3859694004058838,-0.0921940729022026,0,3,16,7,4,2,-1.,16,7,2,1,2.,18,8,2,1,2.,1.0793360415846109e-003,-0.1119038984179497,0.3112520873546600,0,2,5,17,1,3,-1.,5,18,1,1,3.,7.3285802500322461e-005,-0.2023991048336029,0.1558668017387390,0,2,2,0,15,20,-1.,2,10,15,10,2.,0.1367873996496201,-0.2167285978794098,0.1442039012908936,0,3,8,11,6,4,-1.,8,11,3,2,2.,11,13,3,2,2.,-0.0117292599752545,0.4350377023220062,-0.0748865306377411,0,2,8,16,4,3,-1.,8,17,4,1,3.,3.9230841211974621e-003,-0.0502893291413784,0.5883116126060486,0,3,8,18,2,2,-1.,8,18,1,1,2.,9,19,1,1,2.,-2.9819121118634939e-004,-0.3823240101337433,0.0924511328339577,0,2,2,16,13,3,-1.,2,17,13,1,3.,-4.7992770560085773e-003,0.4848878979682922,-0.0731365233659744,0,3,16,16,2,2,-1.,16,16,1,1,2.,17,17,1,1,2.,-3.0155890271998942e-004,-0.3575735986232758,0.1058188006281853,0,2,8,1,6,3,-1.,10,1,2,3,3.,0.0103907696902752,0.0529204681515694,-0.5724965929985046,0,3,16,7,2,2,-1.,16,7,1,1,2.,17,8,1,1,2.,-9.4488041941076517e-004,0.4496682882308960,-0.0830755233764648,0,3,14,7,4,2,-1.,14,7,2,1,2.,16,8,2,1,2.,1.2651870492845774e-003,-0.0966954380273819,0.3130227029323578,0,2,4,0,14,1,-1.,11,0,7,1,2.,0.0170945394784212,-0.0812489762902260,0.3611383140087128,0,3,10,4,8,2,-1.,10,4,4,1,2.,14,5,4,1,2.,2.5973359588533640e-003,-0.1133835017681122,0.2223394960165024,0,2,8,2,3,2,-1.,9,2,1,2,3.,1.4527440071105957e-003,0.0697504431009293,-0.3672071099281311,0,2,12,11,6,3,-1.,12,12,6,1,3.,4.7638658434152603e-003,-0.0657889619469643,0.3832854032516480,0,2,1,5,1,4,-1.,1,7,1,2,2.,-6.2501081265509129e-003,-0.7075446844100952,0.0383501984179020,0,2,1,1,1,18,-1.,1,7,1,6,3.,-3.1765329185873270e-003,0.1375540047883987,-0.2324002981185913,0,2,11,13,3,2,-1.,11,14,3,1,2.,3.2191169448196888e-003,-0.1293545067310333,0.2273788005113602,0,3,0,1,12,2,-1.,0,1,6,1,2.,6,2,6,1,2.,-5.6365579366683960e-003,0.3806715011596680,-0.0672468394041061,0,3,10,18,2,2,-1.,10,18,1,1,2.,11,19,1,1,2.,-2.3844049428589642e-004,-0.3112238049507141,0.0838383585214615,0,3,4,5,4,4,-1.,4,5,2,2,2.,6,7,2,2,2.,-4.1017560288310051e-003,0.2606728076934815,-0.1044974029064179,0,2,6,7,1,3,-1.,6,8,1,1,3.,1.3336989795789123e-003,-0.0582501403987408,0.4768244028091431,0,2,14,10,6,2,-1.,16,10,2,2,3.,-1.2090239906683564e-003,0.1483450978994370,-0.1732946932315826,-1.1188739538192749,72,0,2,16,8,3,6,-1.,17,8,1,6,3.,-3.1760931015014648e-003,0.3333333134651184,-0.1664234995841980,0,2,4,10,6,2,-1.,6,10,2,2,3.,0.0248580798506737,-0.0727288722991943,0.5667458176612854,0,2,6,5,3,7,-1.,7,5,1,7,3.,-7.7597280032932758e-003,0.4625856876373291,-0.0931121781468391,0,2,0,13,6,6,-1.,0,16,6,3,2.,7.8239021822810173e-003,-0.2741461098194122,0.1324304938316345,0,2,12,5,1,9,-1.,12,8,1,3,3.,-0.0109488395974040,0.2234548032283783,-0.1496544927358627,0,2,5,9,3,3,-1.,6,9,1,3,3.,-3.4349008928984404e-003,0.3872498869895935,-0.0661217272281647,0,2,7,5,6,13,-1.,9,5,2,13,3.,-0.0311562903225422,0.2407827973365784,-0.1140690967440605,0,2,19,8,1,10,-1.,19,13,1,5,2.,1.1100519914180040e-003,-0.2820797860622406,0.1327542960643768,0,2,11,18,6,1,-1.,13,18,2,1,3.,3.1762740109115839e-003,0.0345859304070473,-0.5137431025505066,0,2,9,7,6,12,-1.,11,7,2,12,3.,-0.0279774591326714,0.2392677962779999,-0.1325591951608658,0,2,12,7,6,6,-1.,14,7,2,6,3.,-0.0230979397892952,0.3901962041854858,-0.0784780085086823,0,2,15,8,3,4,-1.,16,8,1,4,3.,-3.9731930010020733e-003,0.3069106936454773,-0.0706014037132263,0,2,6,11,4,2,-1.,6,12,4,1,2.,3.0335749033838511e-003,-0.1400219053030014,0.1913485974073410,0,2,1,6,6,8,-1.,3,6,2,8,3.,-0.0108443703502417,0.1654873043298721,-0.1565777957439423,0,2,11,15,6,5,-1.,13,15,2,5,3.,-0.0181505102664232,-0.6324359178543091,0.0395618192851543,0,2,15,17,4,2,-1.,15,18,4,1,2.,7.1052298881113529e-004,-0.1851557046175003,0.1340880990028381,0,2,13,11,6,1,-1.,15,11,2,1,3.,0.0108933402225375,-0.0267302300781012,0.6097180247306824,0,3,5,18,2,2,-1.,5,18,1,1,2.,6,19,1,1,2.,-2.8780900174751878e-004,-0.3006514012813568,0.0731714591383934,0,3,4,8,4,4,-1.,4,8,2,2,2.,6,10,2,2,2.,-3.5855069290846586e-003,0.2621760964393616,-0.0797140970826149,0,2,11,7,9,3,-1.,11,8,9,1,3.,-0.0197592806071043,-0.5903922915458679,0.0406989715993404,0,3,0,3,10,4,-1.,0,3,5,2,2.,5,5,5,2,2.,-0.0108452104032040,0.1636455953121185,-0.1258606016635895,0,2,7,18,6,1,-1.,9,18,2,1,3.,-4.3183090165257454e-003,-0.5747488141059876,0.0376443117856979,0,2,0,8,3,3,-1.,0,9,3,1,3.,1.4913700288161635e-003,0.0609134696424007,-0.3022292852401733,0,3,0,0,6,8,-1.,0,0,3,4,2.,3,4,3,4,2.,0.0156756993383169,-0.0731459110975266,0.2937945127487183,0,2,7,6,3,8,-1.,8,6,1,8,3.,-0.0110335601493716,0.3931880891323090,-0.0470843203365803,0,2,13,7,7,3,-1.,13,8,7,1,3.,8.8555756956338882e-003,0.0376013815402985,-0.4910849034786224,0,2,3,3,2,2,-1.,3,4,2,1,2.,-8.9665671112015843e-004,0.1795202046632767,-0.1108623966574669,0,2,0,3,3,3,-1.,0,4,3,1,3.,-3.0592409893870354e-003,-0.4442946016788483,0.0510054305195808,0,2,9,3,5,2,-1.,9,4,5,1,2.,6.3201179727911949e-003,-0.0528410896658897,0.3719710111618042,0,2,6,5,9,4,-1.,9,5,3,4,3.,0.0206828303635120,0.0576671697199345,-0.3690159916877747,0,2,3,10,12,3,-1.,7,10,4,3,3.,0.0998226627707481,-0.0373770184814930,0.5816559195518494,0,2,8,7,3,6,-1.,9,7,1,6,3.,-6.5854229032993317e-003,0.2850944101810455,-0.0609780699014664,0,2,5,5,6,5,-1.,8,5,3,5,2.,-0.0609003007411957,-0.5103176832199097,0.0377874001860619,0,2,0,5,2,3,-1.,0,6,2,1,3.,-2.9991709161549807e-003,-0.4794301092624664,0.0388338901102543,0,2,9,7,3,4,-1.,10,7,1,4,3.,-9.8906438797712326e-003,0.4060907959938049,-0.0478696487843990,0,2,1,0,6,15,-1.,3,0,2,15,3.,-0.0826889276504517,-0.7067118287086487,0.0274877492338419,0,2,15,1,3,5,-1.,16,1,1,5,3.,5.0060399807989597e-003,0.0282084401696920,-0.5290969014167786,0,2,9,2,3,10,-1.,10,2,1,10,3.,6.1695030890405178e-003,-0.0545548610389233,0.3283798098564148,0,2,8,8,6,12,-1.,10,8,2,12,3.,-3.3914761152118444e-003,0.0921176671981812,-0.2163711041212082,0,2,16,4,3,4,-1.,16,6,3,2,2.,-2.6131230406463146e-003,0.1365101933479309,-0.1378113031387329,0,3,16,7,2,2,-1.,16,7,1,1,2.,17,8,1,1,2.,8.0490659456700087e-004,-0.0686371102929115,0.3358106911182404,0,2,13,0,6,9,-1.,13,3,6,3,3.,-0.0381065085530281,0.2944543063640595,-0.0682392269372940,0,2,7,17,1,3,-1.,7,18,1,1,3.,7.2450799052603543e-005,-0.1675013005733490,0.1217823028564453,0,2,12,1,4,2,-1.,12,2,4,1,2.,1.5837959945201874e-003,-0.0920428484678268,0.2134899049997330,0,2,17,3,1,3,-1.,17,4,1,1,3.,1.2924340553581715e-003,0.0629172325134277,-0.3617450892925263,0,2,0,16,9,3,-1.,0,17,9,1,3.,9.9146775901317596e-003,0.0195340607315302,-0.8101503849029541,0,3,3,6,2,4,-1.,3,6,1,2,2.,4,8,1,2,2.,-1.7086310544982553e-003,0.2552523910999298,-0.0682294592261314,0,2,13,18,3,1,-1.,14,18,1,1,3.,2.1844399161636829e-003,0.0233140494674444,-0.8429678082466126,0,2,0,18,4,2,-1.,2,18,2,2,2.,-3.4244330599904060e-003,0.2721368968486786,-0.0763952285051346,0,2,1,19,2,1,-1.,2,19,1,1,2.,2.7591470279730856e-004,-0.1074284017086029,0.2288897037506104,0,2,0,18,4,2,-1.,0,19,4,1,2.,-6.0005177510902286e-004,-0.2985421121120453,0.0634797364473343,0,2,2,17,1,3,-1.,2,18,1,1,3.,-2.5001438916660845e-004,-0.2717896997928619,0.0696150064468384,0,2,4,8,3,5,-1.,5,8,1,5,3.,6.8751391954720020e-003,-0.0571858994662762,0.3669595122337341,0,2,2,1,6,7,-1.,4,1,2,7,3.,0.0127619002014399,0.0679556876420975,-0.2853415012359619,0,3,3,6,2,8,-1.,3,6,1,4,2.,4,10,1,4,2.,-1.4752789866179228e-003,0.2068066000938416,-0.1005939021706581,0,2,4,5,11,10,-1.,4,10,11,5,2.,0.1213881969451904,-0.0971267968416214,0.1978961974382401,0,2,0,13,20,2,-1.,10,13,10,2,2.,-0.0500812791287899,0.2841717898845673,-0.0678799971938133,0,2,1,13,16,3,-1.,9,13,8,3,2.,0.0314549505710602,-0.0894686728715897,0.2129842042922974,0,3,16,4,4,4,-1.,16,4,2,2,2.,18,6,2,2,2.,1.8878319533541799e-003,-0.1165644004940987,0.1666352003812790,0,3,16,0,4,12,-1.,16,0,2,6,2.,18,6,2,6,2.,-5.7211960665881634e-003,0.2370214015245438,-0.0907766073942184,0,2,14,15,3,1,-1.,15,15,1,1,3.,-1.8076719425152987e-004,0.1795192956924439,-0.1079348027706146,0,2,3,4,12,10,-1.,3,9,12,5,2.,-0.1976184993982315,0.4567429125308991,-0.0404801592230797,0,3,9,18,2,2,-1.,9,18,1,1,2.,10,19,1,1,2.,-2.3846809926908463e-004,-0.2373300939798355,0.0759221613407135,0,3,9,18,2,2,-1.,9,18,1,1,2.,10,19,1,1,2.,2.1540730085689574e-004,0.0816880166530609,-0.2868503034114838,0,3,13,4,2,14,-1.,13,4,1,7,2.,14,11,1,7,2.,0.0101630901917815,-0.0412500202655792,0.4803834855556488,0,2,4,2,6,4,-1.,7,2,3,4,2.,-7.2184870950877666e-003,0.1745858043432236,-0.1014650017023087,0,3,0,0,18,20,-1.,0,0,9,10,2.,9,10,9,10,2.,0.2426317036151886,0.0534264817833900,-0.3231852948665619,0,2,15,11,1,2,-1.,15,12,1,1,2.,6.9304101634770632e-004,-0.1149917989969254,0.1479393988847733,0,3,16,10,2,4,-1.,16,10,1,2,2.,17,12,1,2,2.,3.5475199110805988e-003,-0.0394249781966209,0.5312618017196655,0,3,18,17,2,2,-1.,18,17,1,1,2.,19,18,1,1,2.,2.1403690334409475e-004,0.0697538331151009,-0.2731958031654358,0,2,9,17,1,2,-1.,9,18,1,1,2.,-5.7119462871924043e-004,0.3436990082263947,-0.0576990097761154,0,2,8,4,9,6,-1.,11,4,3,6,3.,-6.6290069371461868e-003,0.1175848990678787,-0.1502013951539993,-1.0888810157775879,66,0,2,6,9,9,10,-1.,9,9,3,10,3.,-0.0265134498476982,0.2056864053010941,-0.2647390067577362,0,2,5,0,5,4,-1.,5,2,5,2,2.,9.7727458924055099e-003,-0.1119284033775330,0.3257054984569550,0,2,5,7,11,4,-1.,5,9,11,2,2.,0.0322903506457806,-0.0985747575759888,0.3177917003631592,0,2,2,4,2,14,-1.,3,4,1,14,2.,-2.8103240765631199e-003,0.1521389931440353,-0.1968640983104706,0,2,8,6,3,5,-1.,9,6,1,5,3.,-0.0109914299100637,0.5140765905380249,-0.0437072105705738,0,2,8,4,3,9,-1.,9,4,1,9,3.,6.3133831135928631e-003,-0.0927810221910477,0.3470247089862824,0,2,0,8,20,6,-1.,0,10,20,2,3.,0.0871059820055962,0.0300536490976810,-0.8281481862068176,0,2,14,16,6,1,-1.,17,16,3,1,2.,1.1799359926953912e-003,-0.1292842030525208,0.2064612060785294,0,2,17,18,2,2,-1.,17,19,2,1,2.,-9.3056890182197094e-004,-0.5002143979072571,0.0936669930815697,0,2,8,17,6,3,-1.,10,17,2,3,3.,-0.0136871701106429,-0.7935814857482910,-6.6733639687299728e-003,0,2,4,1,9,15,-1.,7,1,3,15,3.,-0.0759174525737762,0.3046964108943939,-0.0796558931469917,0,2,11,5,3,12,-1.,12,5,1,12,3.,-2.8559709899127483e-003,0.2096146047115326,-0.1273255050182343,0,2,0,15,4,3,-1.,0,16,4,1,3.,-4.0231510065495968e-003,-0.6581727862358093,0.0506836399435997,0,2,0,0,15,1,-1.,5,0,5,1,3.,0.0175580400973558,-0.0853826925158501,0.3617455959320068,0,2,6,0,6,4,-1.,8,0,2,4,3.,0.0219882391393185,0.0629436969757080,-0.7089633941650391,0,2,2,0,9,3,-1.,5,0,3,3,3.,-2.8599589131772518e-003,0.1468378007411957,-0.1646597981452942,0,2,13,6,3,7,-1.,14,6,1,7,3.,-0.0100308498367667,0.4957993924617767,-0.0271883402019739,0,2,7,6,4,2,-1.,7,7,4,1,2.,-6.9560329429805279e-003,0.2797777950763702,-0.0779533311724663,0,2,6,18,6,1,-1.,8,18,2,1,3.,-3.8356808945536613e-003,-0.5816398262977600,0.0357399396598339,0,2,18,6,2,2,-1.,18,7,2,1,2.,-3.2647319603711367e-003,-0.4994508028030396,0.0469864904880524,0,2,6,4,7,3,-1.,6,5,7,1,3.,-7.8412350267171860e-003,0.3453283011913300,-0.0688104033470154,0,2,12,7,3,1,-1.,13,7,1,1,3.,-8.1718113506212831e-005,0.1504171043634415,-0.1414667963981628,0,3,15,1,2,10,-1.,15,1,1,5,2.,16,6,1,5,2.,-3.2448628917336464e-003,0.2272451072931290,-0.0928602069616318,0,2,0,18,2,2,-1.,0,19,2,1,2.,-7.8561151167377830e-004,-0.4431901872158051,0.0578124411404133,0,2,19,4,1,8,-1.,19,8,1,4,2.,-6.2474247533828020e-004,0.1395238935947418,-0.1466871947050095,0,2,1,17,1,3,-1.,1,18,1,1,3.,-3.2942948746494949e-004,-0.2990157008171082,0.0760667398571968,0,3,0,15,6,4,-1.,0,15,3,2,2.,3,17,3,2,2.,1.2605739757418633e-003,-0.1612560003995895,0.1395380049943924,0,2,19,0,1,18,-1.,19,6,1,6,3.,-0.0516670197248459,-0.5314283967018127,0.0407195203006268,0,2,10,2,6,2,-1.,12,2,2,2,3.,-0.0152856195345521,-0.7820637822151184,0.0271837692707777,0,2,2,8,12,2,-1.,6,8,4,2,3.,0.0690298229455948,-0.0364270210266113,0.7110251784324646,0,2,16,0,4,1,-1.,18,0,2,1,2.,1.4522749697789550e-003,-0.0968905165791512,0.2166842073202133,0,2,8,4,2,6,-1.,8,7,2,3,2.,-2.4765590205788612e-003,0.1164531037211418,-0.1822797954082489,0,2,14,5,2,10,-1.,15,5,1,10,2.,-1.5134819550439715e-003,0.1786397993564606,-0.1221496984362602,0,2,13,4,2,2,-1.,13,5,2,1,2.,-1.5099470037966967e-003,0.1808623969554901,-0.1144606992602348,0,2,11,1,3,6,-1.,11,3,3,2,3.,-6.7054620012640953e-003,0.2510659992694855,-0.0918714627623558,0,2,6,9,12,2,-1.,10,9,4,2,3.,-0.0140752000734210,0.1370750963687897,-0.1733350008726120,0,2,9,16,4,2,-1.,9,17,4,1,2.,-2.2400720044970512e-003,0.4009298086166382,-0.0475768782198429,0,2,5,14,15,4,-1.,5,16,15,2,2.,0.0197823699563742,-0.1904035061597824,0.1492341011762619,0,2,18,16,2,2,-1.,18,17,2,1,2.,2.6002870872616768e-003,0.0469717681407928,-0.4330765902996063,0,3,16,18,2,2,-1.,16,18,1,1,2.,17,19,1,1,2.,-5.3445628145709634e-004,-0.4374423027038574,0.0415201894938946,0,2,6,4,3,8,-1.,7,4,1,8,3.,-0.0174665097147226,0.6581817269325256,-0.0344474911689758,0,2,5,9,3,1,-1.,6,9,1,1,3.,-2.0425589755177498e-003,0.3965792953968048,-0.0440524294972420,0,2,0,8,1,6,-1.,0,10,1,2,3.,2.6661779265850782e-003,0.0587709583342075,-0.3280636966228485,0,2,11,2,9,6,-1.,14,2,3,6,3.,-0.0559823699295521,-0.5173547267913818,0.0357918404042721,0,2,12,2,6,4,-1.,14,2,2,4,3.,-1.5066330088302493e-003,0.1512386947870255,-0.1252018064260483,0,2,1,7,2,4,-1.,1,9,2,2,2.,-0.0114723695442081,-0.6293053030967712,0.0347043313086033,0,2,13,1,6,4,-1.,13,3,6,2,2.,0.0234096292406321,-0.0580633506178856,0.3866822123527527,0,3,4,10,2,10,-1.,4,10,1,5,2.,5,15,1,5,2.,-2.3243729956448078e-003,0.1875409930944443,-0.0983946695923805,0,2,2,16,9,3,-1.,5,16,3,3,3.,-0.0290392991155386,-0.5448690056800842,0.0409263409674168,0,2,1,2,3,9,-1.,2,2,1,9,3.,-0.0144746499136090,-0.6724839210510254,0.0231288503855467,0,2,19,7,1,4,-1.,19,9,1,2,2.,-5.2086091600358486e-003,-0.4327144026756287,0.0437806509435177,0,3,14,11,6,8,-1.,14,11,3,4,2.,17,15,3,4,2.,4.9382899887859821e-003,-0.1087862029671669,0.1934258937835693,0,3,15,12,4,6,-1.,15,12,2,3,2.,17,15,2,3,2.,-4.3193930760025978e-003,0.2408093065023422,-0.1038080006837845,0,3,16,15,2,2,-1.,16,15,1,1,2.,17,16,1,1,2.,2.3705669445917010e-004,-0.0873490720987320,0.2046623975038528,0,3,17,16,2,2,-1.,17,16,1,1,2.,18,17,1,1,2.,4.7858079778961837e-004,0.0456245802342892,-0.3885467052459717,0,3,17,16,2,2,-1.,17,16,1,1,2.,18,17,1,1,2.,-8.5342838428914547e-004,-0.5507794022560120,0.0358258895576000,0,3,2,3,2,2,-1.,2,3,1,1,2.,3,4,1,1,2.,5.4772121075075120e-005,-0.1122523993253708,0.1750351935625076,0,2,10,10,3,3,-1.,11,10,1,3,3.,-3.8445889949798584e-003,0.2452670037746429,-0.0811325684189796,0,2,5,9,7,8,-1.,5,13,7,4,2.,-0.0401284582912922,-0.6312270760536194,0.0269726701080799,0,3,7,16,2,2,-1.,7,16,1,1,2.,8,17,1,1,2.,-1.7886360001284629e-004,0.1985509991645813,-0.1033368036150932,0,3,7,16,2,2,-1.,7,16,1,1,2.,8,17,1,1,2.,1.7668239888735116e-004,-0.0913590118288994,0.1984872072935104,0,2,9,8,10,3,-1.,14,8,5,3,2.,0.0727633833885193,0.0500755794346333,-0.3385263085365295,0,3,6,7,4,8,-1.,6,7,2,4,2.,8,11,2,4,2.,0.0101816300302744,-0.0932299792766571,0.2005959004163742,0,2,1,6,4,3,-1.,1,7,4,1,3.,2.4409969337284565e-003,0.0646366328001022,-0.2692174017429352,0,2,6,10,6,10,-1.,8,10,2,10,3.,-3.6227488890290260e-003,0.1316989064216614,-0.1251484006643295,0,2,4,6,3,6,-1.,5,6,1,6,3.,-1.3635610230267048e-003,0.1635046005249023,-0.1066593974828720,-1.0408929586410522,69,0,3,3,10,4,4,-1.,3,10,2,2,2.,5,12,2,2,2.,-9.6991164609789848e-003,0.6112532019615173,-0.0662253126502037,0,3,3,10,4,4,-1.,3,10,2,2,2.,5,12,2,2,2.,-9.6426531672477722e-003,-1.,2.7699959464371204e-003,0,3,3,10,4,4,-1.,3,10,2,2,2.,5,12,2,2,2.,-9.6381865441799164e-003,1.,-2.9904270195402205e-004,0,2,14,8,2,6,-1.,15,8,1,6,2.,-4.2553939856588840e-003,0.2846438884735107,-0.1554012000560761,0,3,3,10,4,4,-1.,3,10,2,2,2.,5,12,2,2,2.,-9.6223521977663040e-003,-1.,0.0439991801977158,0,3,3,10,4,4,-1.,3,10,2,2,2.,5,12,2,2,2.,-9.1231241822242737e-003,0.8686934113502502,-2.7267890982329845e-003,0,2,12,4,3,9,-1.,13,4,1,9,3.,-8.6240433156490326e-003,0.4535248875617981,-0.0860713794827461,0,2,12,3,1,12,-1.,12,7,1,4,3.,-8.9324144646525383e-003,0.1337555944919586,-0.2601251900196075,0,2,2,0,18,1,-1.,8,0,6,1,3.,-0.0142078101634979,0.3207764029502869,-0.0972264111042023,0,3,10,0,10,6,-1.,10,0,5,3,2.,15,3,5,3,2.,0.0259110108017921,-0.1296408027410507,0.2621864974498749,0,2,18,16,2,2,-1.,18,17,2,1,2.,2.0531509653665125e-004,-0.1240428015589714,0.2106295973062515,0,3,3,5,4,2,-1.,3,5,2,1,2.,5,6,2,1,2.,-5.4795680625829846e-005,0.1197429969906807,-0.2320127934217453,0,2,11,8,3,3,-1.,12,8,1,3,3.,6.8555199541151524e-003,-0.0632761269807816,0.4104425013065338,0,2,11,7,3,5,-1.,12,7,1,5,3.,-0.0122530404478312,0.5488333106040955,-0.0397311002016068,0,2,3,19,15,1,-1.,8,19,5,1,3.,-3.9058770053088665e-003,0.2419098019599915,-0.0970960110425949,0,2,8,13,3,2,-1.,8,14,3,1,2.,2.7560980524867773e-003,-0.1256967931985855,0.1945665031671524,0,3,2,12,8,4,-1.,2,12,4,2,2.,6,14,4,2,2.,-7.7662160620093346e-003,0.2976570129394531,-0.0968181565403938,0,3,16,16,2,2,-1.,16,16,1,1,2.,17,17,1,1,2.,3.8997188676148653e-004,0.0621884018182755,-0.4204089939594269,0,2,7,0,3,2,-1.,8,0,1,2,3.,3.3579880837351084e-003,0.0474981404840946,-0.6321688294410706,0,2,6,7,2,5,-1.,7,7,1,5,2.,-0.0167455393821001,0.7109813094139099,-0.0391573496162891,0,2,18,0,2,17,-1.,19,0,1,17,2.,-6.5409899689257145e-003,-0.3504317104816437,0.0706169530749321,0,2,16,16,1,3,-1.,16,17,1,1,3.,3.0016340315341949e-004,0.0919024571776390,-0.2461867034435272,0,2,14,8,3,7,-1.,15,8,1,7,3.,0.0149189904332161,-0.0519094504415989,0.5663604140281677,0,3,10,17,2,2,-1.,10,17,1,1,2.,11,18,1,1,2.,4.8153079114854336e-004,0.0646595582365990,-0.3659060895442963,0,2,4,9,1,3,-1.,4,10,1,1,3.,-3.0211321427486837e-004,0.1792656928300858,-0.1141066029667854,0,2,18,10,2,3,-1.,18,11,2,1,3.,3.8521419628523290e-004,0.1034561991691589,-0.2007246017456055,0,2,12,1,3,10,-1.,13,1,1,10,3.,8.0837132409214973e-003,-0.0660734623670578,0.3028424978256226,0,2,8,12,9,1,-1.,11,12,3,1,3.,-0.0228049699217081,0.5296235084533691,-0.0401189997792244,0,3,5,18,2,2,-1.,5,18,1,1,2.,6,19,1,1,2.,1.9440450705587864e-004,0.0818548202514648,-0.2466336041688919,0,2,19,6,1,9,-1.,19,9,1,3,3.,-0.0128480903804302,-0.3497331142425537,0.0569162294268608,0,3,4,7,2,4,-1.,4,7,1,2,2.,5,9,1,2,2.,-1.0937290498986840e-003,0.2336868047714233,-0.0916048064827919,0,2,1,4,6,14,-1.,3,4,2,14,3.,1.0032650316134095e-003,0.1185218021273613,-0.1846919059753418,0,2,10,5,9,3,-1.,13,5,3,3,3.,-0.0446884296834469,-0.6436246037483215,0.0303632691502571,0,2,18,7,2,6,-1.,18,9,2,2,3.,8.1657543778419495e-003,0.0436746589839458,-0.4300208985805512,0,2,5,6,2,7,-1.,6,6,1,7,2.,-0.0117178102955222,0.4178147912025452,-0.0482336990535259,0,2,10,4,6,8,-1.,13,4,3,8,2.,0.0842771306633949,0.0534612797200680,-0.3795219063758850,0,2,0,8,2,9,-1.,0,11,2,3,3.,0.0142118399962783,0.0449009388685226,-0.4298149943351746,0,2,0,7,5,3,-1.,0,8,5,1,3.,1.5028340276330709e-003,0.0822276398539543,-0.2470639944076538,0,2,8,1,7,2,-1.,8,2,7,1,2.,0.0100035797804594,-0.0572216697037220,0.3460937142372131,0,2,7,5,3,5,-1.,8,5,1,5,3.,-9.0706320479512215e-003,0.4505808949470520,-0.0427953191101551,0,2,19,2,1,2,-1.,19,3,1,1,2.,-3.3141620224341750e-004,0.1833691000938416,-0.1075994968414307,0,2,6,7,10,11,-1.,11,7,5,11,2.,0.1972327977418900,-0.0303638298064470,0.6642342805862427,0,2,9,19,6,1,-1.,11,19,2,1,3.,-7.1258801035583019e-003,-0.8922504782676697,0.0256699901074171,0,2,3,0,12,1,-1.,7,0,4,1,3.,8.6921341717243195e-003,-0.0707643702626228,0.2821052968502045,0,2,4,1,6,5,-1.,6,1,2,5,3.,8.9262127876281738e-003,0.0710782334208488,-0.3023256063461304,0,2,6,12,12,6,-1.,10,12,4,6,3.,0.0572860091924667,0.0509741306304932,-0.3919695019721985,0,2,16,13,2,3,-1.,16,14,2,1,3.,3.7920880131423473e-003,0.0338419415056705,-0.5101628899574280,0,2,7,14,4,2,-1.,7,15,4,1,2.,-1.4508679741993546e-003,0.3087914884090424,-0.0638450831174850,0,2,7,14,2,2,-1.,7,15,2,1,2.,9.8390132188796997e-004,-0.1302956938743591,0.1460441052913666,0,3,3,10,2,4,-1.,3,10,1,2,2.,4,12,1,2,2.,-1.7221809830516577e-003,0.2915700972080231,-0.0685495585203171,0,2,0,3,2,6,-1.,0,5,2,2,3.,0.0109482500702143,0.0343514084815979,-0.4770225882530212,0,3,1,10,2,2,-1.,1,10,1,1,2.,2,11,1,1,2.,-1.7176309484057128e-005,0.1605526953935623,-0.1169084012508392,0,2,16,4,4,3,-1.,16,5,4,1,3.,-5.4884208366274834e-003,-0.4341588914394379,0.0461062416434288,0,3,5,10,2,4,-1.,5,10,1,2,2.,6,12,1,2,2.,-3.0975250992923975e-003,0.3794333934783936,-0.0568605512380600,0,2,5,11,13,2,-1.,5,12,13,1,2.,6.4182081259787083e-003,-0.1585821062326431,0.1233541965484619,0,2,10,2,3,11,-1.,11,2,1,11,3.,0.0118312397971749,-0.0409292913973331,0.4587895870208740,0,2,10,2,4,4,-1.,10,4,4,2,2.,0.0135404998436570,-0.0537255592644215,0.3505612015724182,0,2,8,8,6,2,-1.,10,8,2,2,3.,-2.5932150892913342e-003,0.1101052016019821,-0.1675221025943756,0,2,11,2,3,3,-1.,12,2,1,3,3.,1.6856270376592875e-003,0.0665743574500084,-0.3083502054214478,0,3,6,18,14,2,-1.,6,18,7,1,2.,13,19,7,1,2.,2.6524690911173820e-003,0.0663184821605682,-0.2786133885383606,0,2,17,7,1,12,-1.,17,11,1,4,3.,-7.7341729775071144e-003,0.1971835941076279,-0.1078291982412338,0,2,10,5,10,3,-1.,10,6,10,1,3.,5.0944271497428417e-003,0.0853374898433685,-0.2484700977802277,0,2,6,1,3,3,-1.,7,1,1,3,3.,-2.9162371065467596e-003,-0.4747635126113892,0.0335664898157120,0,2,13,8,3,1,-1.,14,8,1,1,3.,3.0121419113129377e-003,-0.0475753806531429,0.4258680045604706,0,2,10,14,2,6,-1.,10,16,2,2,3.,3.1694869976490736e-003,-0.1051945015788078,0.1716345995664597,0,2,4,1,12,14,-1.,8,1,4,14,3.,0.2232756018638611,-0.0143702095374465,0.9248365163803101,0,2,14,1,6,14,-1.,16,1,2,14,3.,-0.0955850481987000,-0.7420663833618164,0.0278189703822136,0,3,3,16,2,2,-1.,3,16,1,1,2.,4,17,1,1,2.,3.4773729566950351e-005,-0.1276578009128571,0.1292666941881180,0,2,0,16,2,2,-1.,0,17,2,1,2.,7.2459770308341831e-005,-0.1651857942342758,0.1003680974245071,-1.0566600561141968,59,0,3,15,6,4,6,-1.,15,6,2,3,2.,17,9,2,3,2.,-6.5778270363807678e-003,0.3381525874137878,-0.1528190970420837,0,2,12,5,2,2,-1.,12,6,2,1,2.,-1.0922809597104788e-003,0.2228236943483353,-0.1930849999189377,0,2,7,6,6,13,-1.,9,6,2,13,3.,-0.0297595895826817,0.2595987021923065,-0.1540940999984741,0,2,1,9,6,5,-1.,3,9,2,5,3.,-0.0131475403904915,0.1903381049633026,-0.1654399931430817,0,2,0,5,3,4,-1.,0,7,3,2,2.,-1.4396329643204808e-003,0.2007171064615250,-0.1233894005417824,0,3,4,1,16,2,-1.,4,1,8,1,2.,12,2,8,1,2.,-3.5928250290453434e-003,0.2398552000522614,-0.1292214989662170,0,3,1,18,4,2,-1.,1,18,2,1,2.,3,19,2,1,2.,-1.5314699849113822e-003,-0.4901489913463593,0.1027503013610840,0,2,7,7,3,4,-1.,8,7,1,4,3.,-6.2372139655053616e-003,0.3121463954448700,-0.1140562966465950,0,2,3,4,9,3,-1.,6,4,3,3,3.,-0.0333646498620510,-0.4952087998390198,0.0513284504413605,0,2,4,6,6,10,-1.,6,6,2,10,3.,-0.0228276997804642,0.3255882859230042,-0.0650893077254295,0,2,9,0,8,10,-1.,13,0,4,10,2.,-0.0861990973353386,-0.6764633059501648,0.0269856993108988,0,2,8,0,8,1,-1.,12,0,4,1,2.,-2.1065981127321720e-003,0.2245243042707443,-0.1261022984981537,0,3,6,2,8,16,-1.,6,2,4,8,2.,10,10,4,8,2.,0.0391201488673687,0.1132939979434013,-0.2686063051223755,0,3,14,10,2,10,-1.,14,10,1,5,2.,15,15,1,5,2.,3.5082739777863026e-003,-0.1135995984077454,0.2564977109432221,0,2,12,11,1,2,-1.,12,12,1,1,2.,5.9289898490533233e-004,-0.1494296938180924,0.1640983968973160,0,2,16,0,3,8,-1.,17,0,1,8,3.,7.1766850305721164e-004,0.0999056920409203,-0.2196796983480454,0,2,14,0,6,10,-1.,17,0,3,10,2.,-0.0218036007136106,-0.3171172142028809,0.0828895866870880,0,2,16,0,3,5,-1.,17,0,1,5,3.,-3.2962779514491558e-003,-0.3804872930049896,0.0608193799853325,0,2,4,5,11,2,-1.,4,6,11,1,2.,2.4196270387619734e-003,-0.0960130169987679,0.2854058146476746,0,2,1,0,2,1,-1.,2,0,1,1,2.,-4.4187481398694217e-004,0.2212793976068497,-0.0974349081516266,0,2,0,0,2,3,-1.,0,1,2,1,3.,3.4523929934948683e-003,0.0375531204044819,-0.5796905159950256,0,2,11,6,6,11,-1.,13,6,2,11,3.,-0.0218346007168293,0.2956213951110840,-0.0800483003258705,0,2,14,0,3,1,-1.,15,0,1,1,3.,-2.1309500152710825e-004,0.2281450927257538,-0.1011418998241425,0,2,19,7,1,2,-1.,19,8,1,1,2.,-1.6166249988600612e-003,-0.5054119825363159,0.0447645410895348,0,2,17,0,3,9,-1.,18,0,1,9,3.,7.5959609821438789e-003,0.0459865406155586,-0.4119768142700195,0,2,12,7,3,4,-1.,13,7,1,4,3.,3.8601809646934271e-003,-0.0865631699562073,0.2480999976396561,0,3,0,1,14,2,-1.,0,1,7,1,2.,7,2,7,1,2.,6.0622231103479862e-003,-0.0755573734641075,0.2843326032161713,0,2,3,1,3,2,-1.,4,1,1,2,3.,-1.7097420059144497e-003,-0.3529582023620606,0.0584104992449284,0,2,4,0,15,2,-1.,9,0,5,2,3.,0.0165155790746212,-0.0804869532585144,0.2353743016719818,0,2,10,2,6,1,-1.,12,2,2,1,3.,4.8465100117027760e-003,0.0418952181935310,-0.4844304919242859,0,2,9,4,6,11,-1.,11,4,2,11,3.,-0.0311671700328588,0.1919230967760086,-0.1026815995573998,0,2,2,16,2,4,-1.,2,18,2,2,2.,6.1892281519249082e-004,-0.2108577042818070,0.0938869267702103,0,2,6,17,6,3,-1.,8,17,2,3,3.,0.0119463102892041,0.0390961691737175,-0.6224862933158875,0,2,7,9,6,2,-1.,9,9,2,2,3.,-7.5677200220525265e-003,0.1593683958053589,-0.1225078031420708,0,2,6,8,9,2,-1.,9,8,3,2,3.,-0.0537474118173122,-0.5562217831611633,0.0411900095641613,0,3,6,6,2,10,-1.,6,6,1,5,2.,7,11,1,5,2.,0.0155135300010443,-0.0398268811404705,0.6240072846412659,0,2,0,11,2,3,-1.,0,12,2,1,3.,1.5246650436893106e-003,0.0701386779546738,-0.3078907132148743,0,2,11,15,4,1,-1.,13,15,2,1,2.,-4.8315100139006972e-004,0.1788765937089920,-0.1095862016081810,0,2,6,17,1,2,-1.,6,18,1,1,2.,2.7374739293009043e-003,0.0274785906076431,-0.8848956823348999,0,2,0,0,6,20,-1.,2,0,2,20,3.,-0.0657877177000046,-0.4643214046955109,0.0350371487438679,0,2,3,10,2,2,-1.,4,10,1,2,2.,1.2409730115905404e-003,-0.0964792370796204,0.2877922058105469,0,2,4,7,3,5,-1.,5,7,1,5,3.,8.1398809561505914e-004,0.1151171997189522,-0.1676616072654724,0,2,3,12,6,2,-1.,5,12,2,2,3.,0.0239018201828003,-0.0326031893491745,0.6001734733581543,0,2,6,15,7,4,-1.,6,17,7,2,2.,0.0275566000491381,-0.0661373436450958,0.2999447882175446,0,3,17,16,2,2,-1.,17,16,1,1,2.,18,17,1,1,2.,-3.8070970913395286e-004,-0.3388118147850037,0.0644507706165314,0,2,15,1,3,16,-1.,16,1,1,16,3.,-1.3335429830476642e-003,0.1458866000175476,-0.1321762055158615,0,2,6,16,6,3,-1.,8,16,2,3,3.,-9.3507990241050720e-003,-0.5117782950401306,0.0349694713950157,0,2,15,14,3,2,-1.,15,15,3,1,2.,7.6215229928493500e-003,0.0232495293021202,-0.6961941123008728,0,2,12,16,1,2,-1.,12,17,1,1,2.,-5.3407860832521692e-005,0.2372737973928452,-0.0869107097387314,0,3,0,2,4,4,-1.,0,2,2,2,2.,2,4,2,2,2.,-1.5332329785451293e-003,0.1922841072082520,-0.1042239964008331,0,3,1,1,6,4,-1.,1,1,3,2,2.,4,3,3,2,2.,4.3135890737175941e-003,-0.0962195470929146,0.2560121119022369,0,2,1,18,1,2,-1.,1,19,1,1,2.,-2.3042880638968199e-004,-0.3156475126743317,0.0588385984301567,0,2,4,7,2,3,-1.,4,8,2,1,3.,-7.8411828726530075e-003,-0.6634092926979065,0.0245009995996952,0,2,1,0,9,14,-1.,1,7,9,7,2.,0.1710374057292938,0.0338314995169640,-0.4561594128608704,0,3,4,9,2,6,-1.,4,9,1,3,2.,5,12,1,3,2.,-1.6011140542104840e-003,0.2157489061355591,-0.0836225301027298,0,2,3,9,4,3,-1.,5,9,2,3,2.,-0.0105357803404331,0.2455231994390488,-0.0823844894766808,0,2,0,9,2,4,-1.,0,11,2,2,2.,-5.8351638726890087e-003,-0.4780732989311218,0.0440862216055393,0,2,16,6,3,10,-1.,17,6,1,10,3.,-0.0187061093747616,-0.6002402901649475,0.0214100405573845,0,2,16,11,2,1,-1.,17,11,1,1,2.,-9.3307439237833023e-004,0.2432359009981155,-0.0741657167673111,-0.9769343137741089,88,0,2,5,7,4,4,-1.,5,9,4,2,2.,0.0106462296098471,-0.1386138945817947,0.2649407088756561,0,2,10,11,9,2,-1.,13,11,3,2,3.,0.0352982692420483,-0.0758217275142670,0.3902106881141663,0,3,15,10,2,2,-1.,15,10,1,1,2.,16,11,1,1,2.,7.5638387352228165e-004,-0.0955214425921440,0.2906199991703033,0,2,10,6,6,14,-1.,10,13,6,7,2.,0.0924977064132690,-0.2770423889160156,0.0794747024774551,0,2,14,7,3,5,-1.,15,7,1,5,3.,-2.9340879991650581e-003,0.2298953980207443,-0.0785500109195709,0,2,6,11,12,3,-1.,10,11,4,3,3.,-0.0865358486771584,0.4774481058120728,-6.8231220357120037e-003,0,2,17,16,1,2,-1.,17,17,1,1,2.,5.4699288739357144e-005,-0.2264260947704315,0.0881921127438545,0,2,8,5,5,4,-1.,8,7,5,2,2.,-0.0365925207734108,0.2735387086868286,-0.0986067429184914,0,2,11,6,4,2,-1.,11,7,4,1,2.,2.6469118893146515e-003,-0.0440839789807796,0.3144528865814209,0,3,3,4,8,2,-1.,3,4,4,1,2.,7,5,4,1,2.,-4.4271810911595821e-003,0.2382272928953171,-0.0867842733860016,0,2,0,8,6,6,-1.,2,8,2,6,3.,-5.1882481202483177e-003,0.1504276990890503,-0.1267210990190506,0,2,7,4,6,2,-1.,7,5,6,1,2.,4.5530400238931179e-003,-0.0559450201690197,0.3650163114070892,0,2,7,3,6,3,-1.,9,3,2,3,3.,0.0145624103024602,0.0363977700471878,-0.5355919003486633,0,2,2,17,3,3,-1.,2,18,3,1,3.,6.8677567469421774e-005,-0.1747962981462479,0.1106870993971825,0,2,3,10,6,1,-1.,5,10,2,1,3.,-5.9744901955127716e-003,0.3107787072658539,-0.0665302276611328,0,2,7,2,6,2,-1.,9,2,2,2,3.,-5.8691250160336494e-003,-0.3190149068832398,0.0639318302273750,0,2,4,11,9,1,-1.,7,11,3,1,3.,-0.0111403102055192,0.2436479032039642,-0.0809351801872253,0,2,7,7,11,12,-1.,7,13,11,6,2.,-0.0586435310542583,-0.7608326077461243,0.0308096297085285,0,2,3,2,3,4,-1.,4,2,1,4,3.,-4.6097282320261002e-003,-0.4531502127647400,0.0298790596425533,0,2,9,7,9,3,-1.,12,7,3,3,3.,-9.3032103031873703e-003,0.1451337933540344,-0.1103316992521286,0,3,15,11,2,6,-1.,15,11,1,3,2.,16,14,1,3,2.,1.3253629440441728e-003,-0.0976989567279816,0.1964644044637680,0,2,0,5,5,3,-1.,0,6,5,1,3.,4.9800761044025421e-003,0.0336480811238289,-0.3979220986366272,0,2,8,1,6,12,-1.,10,1,2,12,3.,-7.6542161405086517e-003,0.0908419936895370,-0.1596754938364029,0,2,3,7,15,13,-1.,8,7,5,13,3.,-0.3892059028148651,-0.6657109260559082,0.0190288294106722,0,2,0,9,9,9,-1.,0,12,9,3,3.,-0.1001966968178749,-0.5755926966667175,0.0242827795445919,0,2,16,0,3,8,-1.,17,0,1,8,3.,7.3541211895644665e-004,0.0879198014736176,-0.1619534045457840,0,2,16,2,4,2,-1.,18,2,2,2,2.,-3.4802639856934547e-003,0.2606449127197266,-0.0602008104324341,0,2,13,0,6,5,-1.,16,0,3,5,2.,8.4000425413250923e-003,-0.1097972989082336,0.1570730954408646,0,2,15,1,3,2,-1.,16,1,1,2,3.,2.3786011151969433e-003,0.0360582396388054,-0.4727719128131867,0,2,11,8,3,2,-1.,12,8,1,2,3.,7.3831682093441486e-003,-0.0357563607394695,0.4949859082698822,0,3,1,8,2,12,-1.,1,8,1,6,2.,2,14,1,6,2.,3.2115620560944080e-003,-0.1012556031346321,0.1574798971414566,0,2,0,1,6,12,-1.,2,1,2,12,3.,-0.0782096683979034,-0.7662708163261414,0.0229658298194408,0,2,19,17,1,3,-1.,19,18,1,1,3.,5.3303989261621609e-005,-0.1341435015201569,0.1111491993069649,0,2,11,3,3,10,-1.,12,3,1,10,3.,-9.6419155597686768e-003,0.2506802976131439,-0.0666081383824348,0,2,8,1,9,8,-1.,11,1,3,8,3.,-0.0710926726460457,-0.4005681872367859,0.0402977913618088,0,3,18,16,2,2,-1.,18,16,1,1,2.,19,17,1,1,2.,3.5171560011804104e-004,0.0418611802160740,-0.3296119868755341,0,3,18,16,2,2,-1.,18,16,1,1,2.,19,17,1,1,2.,-3.3458150574006140e-004,-0.2602983117103577,0.0678927376866341,0,2,6,13,2,6,-1.,6,15,2,2,3.,-4.1451421566307545e-003,0.2396769970655441,-0.0720933377742767,0,2,9,14,2,2,-1.,9,15,2,1,2.,3.1754500232636929e-003,-0.0712352693080902,0.2412845045328140,0,3,14,10,2,4,-1.,14,10,1,2,2.,15,12,1,2,2.,-5.5184490047395229e-003,0.5032023787498474,-0.0296866800636053,0,3,0,15,2,2,-1.,0,15,1,1,2.,1,16,1,1,2.,-3.0242869979701936e-004,0.2487905025482178,-0.0567585788667202,0,3,6,7,2,2,-1.,6,7,1,1,2.,7,8,1,1,2.,-1.3125919504091144e-003,0.3174780011177063,-0.0418458618223667,0,3,11,18,2,2,-1.,11,18,1,1,2.,12,19,1,1,2.,-2.7123570907860994e-004,-0.2704207003116608,0.0568289905786514,0,3,0,0,6,4,-1.,0,0,3,2,2.,3,2,3,2,2.,-7.3241777718067169e-003,0.2755667865276337,-0.0542529709637165,0,2,4,1,6,6,-1.,6,1,2,6,3.,-0.0168517101556063,-0.3485291004180908,0.0453689992427826,0,2,15,13,5,4,-1.,15,15,5,2,2.,0.0299021005630493,0.0316210798919201,-0.4311437010765076,0,2,7,17,6,1,-1.,9,17,2,1,3.,2.8902660124003887e-003,0.0380299612879753,-0.3702709972858429,0,2,16,19,4,1,-1.,18,19,2,1,2.,-1.9242949783802032e-003,0.2480027973651886,-0.0593332983553410,0,2,16,16,4,4,-1.,18,16,2,4,2.,4.9354149959981441e-003,-0.0830684006214142,0.2204380929470062,0,2,7,8,9,4,-1.,10,8,3,4,3.,0.0820756033062935,-0.0194134395569563,0.6908928751945496,0,3,16,18,2,2,-1.,16,18,1,1,2.,17,19,1,1,2.,-2.4699489586055279e-004,-0.2466056942939758,0.0647764503955841,0,3,2,9,2,4,-1.,2,9,1,2,2.,3,11,1,2,2.,-1.8365769647061825e-003,0.2883616089820862,-0.0533904582262039,0,3,0,3,8,4,-1.,0,3,4,2,2.,4,5,4,2,2.,-4.9553811550140381e-003,0.1274082958698273,-0.1255941987037659,0,2,0,1,8,1,-1.,4,1,4,1,2.,-8.3086621016263962e-003,0.2347811013460159,-0.0716764926910400,0,2,0,5,8,9,-1.,4,5,4,9,2.,-0.1087991967797279,-0.2599223852157593,0.0586897395551205,0,2,7,18,6,2,-1.,9,18,2,2,3.,-9.6786450594663620e-003,-0.7072042822837830,0.0187492594122887,0,2,0,4,1,12,-1.,0,8,1,4,3.,-0.0271368306130171,-0.5838422775268555,0.0216841306537390,0,2,19,13,1,6,-1.,19,15,1,2,3.,-6.5389778465032578e-003,-0.5974891185760498,0.0214803107082844,0,2,2,8,6,8,-1.,4,8,2,8,3.,-0.0120956301689148,0.1326903998851776,-0.0997227206826210,0,2,0,0,9,17,-1.,3,0,3,17,3.,-0.1677609980106354,-0.5665506720542908,0.0321230888366699,0,2,7,9,6,8,-1.,9,9,2,8,3.,-0.0132625503465533,0.1149559020996094,-0.1173838973045349,0,2,5,10,9,4,-1.,8,10,3,4,3.,0.0767445191740990,-0.0314132310450077,0.5993549227714539,0,2,5,0,8,3,-1.,5,1,8,1,3.,5.0785229541361332e-003,-0.0529119409620762,0.2334239929914475,0,3,16,6,4,4,-1.,16,6,2,2,2.,18,8,2,2,2.,3.1800279393792152e-003,-0.0777343884110451,0.1765290945768356,0,3,17,4,2,8,-1.,17,4,1,4,2.,18,8,1,4,2.,-1.7729829996824265e-003,0.1959162950515747,-0.0797521993517876,0,2,2,16,1,3,-1.,2,17,1,1,3.,-4.8560940194875002e-004,-0.2880037128925324,0.0490471199154854,0,2,2,16,1,3,-1.,2,17,1,1,3.,3.6554320831783116e-004,0.0679228976368904,-0.2249943017959595,0,2,11,0,1,3,-1.,11,1,1,1,3.,-2.6938671362586319e-004,0.1658217012882233,-0.0897440984845161,0,2,11,2,9,7,-1.,14,2,3,7,3.,0.0786842331290245,0.0260816793888807,-0.5569373965263367,0,2,10,2,3,6,-1.,11,2,1,6,3.,-7.3774810880422592e-004,0.1403687000274658,-0.1180030032992363,0,2,5,9,15,2,-1.,5,10,15,1,2.,0.0239578299224377,0.0304707400500774,-0.4615997970104218,0,2,8,16,6,2,-1.,8,17,6,1,2.,-1.6239080578088760e-003,0.2632707953453064,-0.0567653700709343,0,3,9,16,10,2,-1.,9,16,5,1,2.,14,17,5,1,2.,-9.0819748584181070e-004,0.1546245962381363,-0.1108706966042519,0,3,9,17,2,2,-1.,9,17,1,1,2.,10,18,1,1,2.,3.9806248969398439e-004,0.0556303709745407,-0.2833195924758911,0,3,10,15,6,4,-1.,10,15,3,2,2.,13,17,3,2,2.,2.0506449509412050e-003,-0.0916048362851143,0.1758553981781006,0,2,4,5,15,12,-1.,9,5,5,12,3.,0.0267425496131182,0.0620030313730240,-0.2448700070381165,0,2,11,13,2,3,-1.,11,14,2,1,3.,-2.1497008856385946e-003,0.2944929897785187,-0.0532181486487389,0,2,8,13,7,3,-1.,8,14,7,1,3.,5.6671658530831337e-003,-0.0642982423305511,0.2490568011999130,0,2,1,12,1,2,-1.,1,13,1,1,2.,6.8317902332637459e-005,-0.1681963056325913,0.0965485796332359,0,3,16,18,2,2,-1.,16,18,1,1,2.,17,19,1,1,2.,1.7600439605303109e-004,0.0653080120682716,-0.2426788061857224,0,2,1,19,18,1,-1.,7,19,6,1,3.,4.1861608624458313e-003,-0.0979885831475258,0.1805288940668106,0,2,1,17,6,1,-1.,4,17,3,1,2.,-2.1808340679854155e-003,0.1923127025365830,-0.0941239297389984,0,2,1,3,1,12,-1.,1,9,1,6,2.,0.0217304006218910,0.0355785116553307,-0.4508853852748871,0,2,0,9,3,6,-1.,0,11,3,2,3.,-0.0147802699357271,-0.4392701089382172,0.0317355915904045,0,2,5,4,3,10,-1.,6,4,1,10,3.,-3.6145891062915325e-003,0.1981147974729538,-0.0777014195919037,0,2,6,17,2,1,-1.,7,17,1,1,2.,1.8892709631472826e-003,0.0199624393135309,-0.7204172015190125,0,2,1,0,6,12,-1.,3,0,2,12,3.,-1.3822480104863644e-003,0.0984669476747513,-0.1488108038902283,0,2,4,7,9,2,-1.,7,7,3,2,3.,-3.9505911991000175e-003,0.1159323006868362,-0.1279197037220001,-1.0129359960556030,58,0,2,6,11,9,1,-1.,9,11,3,1,3.,-0.0193955395370722,0.4747475087642670,-0.1172109022736549,0,2,17,10,2,10,-1.,17,15,2,5,2.,0.0131189199164510,-0.2555212974548340,0.1637880057096481,0,3,4,10,2,10,-1.,4,10,1,5,2.,5,15,1,5,2.,-5.1606801571324468e-004,0.1945261955261231,-0.1744889020919800,0,2,12,3,3,12,-1.,13,3,1,12,3.,-0.0131841599941254,0.4418145120143890,-0.0900487527251244,0,3,15,3,4,6,-1.,15,3,2,3,2.,17,6,2,3,2.,3.4657081123441458e-003,-0.1347709000110626,0.1805634051561356,0,2,12,8,3,3,-1.,13,8,1,3,3.,6.2980200164020061e-003,-0.0541649796068668,0.3603338003158569,0,2,4,14,2,4,-1.,4,16,2,2,2.,1.6879989998415112e-003,-0.1999794989824295,0.1202159970998764,0,2,6,16,1,3,-1.,6,17,1,1,3.,3.6039709812030196e-004,0.1052414029836655,-0.2411606013774872,0,2,1,1,2,3,-1.,2,1,1,3,2.,-1.5276849735528231e-003,0.2813552916049957,-0.0689648166298866,0,2,0,2,4,1,-1.,2,2,2,1,2.,3.5033570602536201e-003,-0.0825195834040642,0.4071359038352966,0,2,8,17,12,3,-1.,12,17,4,3,3.,-4.7337161377072334e-003,0.1972700953483582,-0.1171014010906220,0,2,9,16,6,4,-1.,11,16,2,4,3.,-0.0115571497008204,-0.5606111288070679,0.0681709572672844,0,2,4,6,3,6,-1.,4,9,3,3,2.,-0.0274457205086946,0.4971862137317658,-0.0623801499605179,0,2,6,2,12,9,-1.,6,5,12,3,3.,-0.0528257787227631,0.1692122071981430,-0.1309355050325394,0,3,6,0,14,20,-1.,6,0,7,10,2.,13,10,7,10,2.,-0.2984969913959503,-0.6464967131614685,0.0400768183171749,0,3,15,16,2,2,-1.,15,16,1,1,2.,16,17,1,1,2.,-2.6307269581593573e-004,0.2512794137001038,-0.0894948393106461,0,3,15,16,2,2,-1.,15,16,1,1,2.,16,17,1,1,2.,2.3261709429789335e-004,-0.0868439897894859,0.2383197993040085,0,2,19,8,1,3,-1.,19,9,1,1,3.,2.3631360090803355e-004,0.1155446022748947,-0.1893634945154190,0,2,13,4,1,2,-1.,13,5,1,1,2.,2.0742209162563086e-003,-0.0485948510468006,0.5748599171638489,0,2,0,4,4,2,-1.,0,5,4,1,2.,-7.0308889262378216e-003,-0.5412080883979797,0.0487437509000301,0,2,19,5,1,6,-1.,19,7,1,2,3.,8.2652270793914795e-003,0.0264945197850466,-0.6172845959663391,0,2,16,0,2,1,-1.,17,0,1,1,2.,2.0042760297656059e-004,-0.1176863014698029,0.1633386015892029,0,2,13,1,1,3,-1.,13,2,1,1,3.,1.6470040427520871e-003,-0.0599549189209938,0.3517970144748688,0,2,17,17,1,3,-1.,17,18,1,1,3.,-3.5642538568936288e-004,-0.3442029953002930,0.0649482533335686,0,3,5,4,8,8,-1.,5,4,4,4,2.,9,8,4,4,2.,-0.0309358704835176,0.1997970044612885,-0.0976936966180801,0,3,1,2,2,2,-1.,1,2,1,1,2.,2,3,1,1,2.,-6.3578772824257612e-004,-0.3148139119148254,0.0594250410795212,0,3,0,0,8,6,-1.,0,0,4,3,2.,4,3,4,3,2.,-0.0118621801957488,0.2004369050264359,-0.0894475430250168,0,2,6,3,4,2,-1.,6,4,4,1,2.,7.1508930996060371e-003,-0.0390060618519783,0.5332716107368469,0,2,1,0,3,3,-1.,1,1,3,1,3.,-2.0059191156178713e-003,-0.2846972048282623,0.0707236081361771,0,2,6,1,7,2,-1.,6,2,7,1,2.,3.6412389017641544e-003,-0.1066031977534294,0.2494480013847351,0,2,2,6,12,6,-1.,6,6,4,6,3.,-0.1346742957830429,0.4991008043289185,-0.0403322204947472,0,2,1,16,9,2,-1.,4,16,3,2,3.,-2.2547659464180470e-003,0.1685169041156769,-0.1111928001046181,0,2,7,15,6,4,-1.,9,15,2,4,3.,4.3842289596796036e-003,0.0861394926905632,-0.2743177115917206,0,2,6,15,12,1,-1.,12,15,6,1,2.,-7.3361168615520000e-003,0.2487521022558212,-0.0959191620349884,0,2,17,17,1,3,-1.,17,18,1,1,3.,6.4666912658140063e-004,0.0674315765500069,-0.3375408053398132,0,3,17,15,2,2,-1.,17,15,1,1,2.,18,16,1,1,2.,2.2983769304119051e-004,-0.0839030519127846,0.2458409965038300,0,2,3,13,3,3,-1.,3,14,3,1,3.,6.7039071582257748e-003,0.0290793292224407,-0.6905593872070313,0,2,10,17,1,3,-1.,10,18,1,1,3.,5.0734888645820320e-005,-0.1569671928882599,0.1196542978286743,0,2,4,0,14,8,-1.,11,0,7,8,2.,-0.2033555954694748,-0.6950634717941284,0.0275075193494558,0,2,2,0,12,2,-1.,6,0,4,2,3.,9.4939414411783218e-003,-0.0874493718147278,0.2396833002567291,0,2,2,0,4,3,-1.,4,0,2,3,2.,-2.4055240210145712e-003,0.2115096002817154,-0.1314893066883087,0,2,13,1,1,2,-1.,13,2,1,1,2.,-1.1342419747961685e-004,0.1523378938436508,-0.1272590011358261,0,2,7,5,3,6,-1.,8,5,1,6,3.,0.0149922100827098,-0.0341279692947865,0.5062407255172730,0,3,18,2,2,2,-1.,18,2,1,1,2.,19,3,1,1,2.,7.4068200774490833e-004,0.0487647503614426,-0.4022532105445862,0,2,15,1,2,14,-1.,16,1,1,14,2.,-4.2459447868168354e-003,0.2155476063489914,-0.0871269926428795,0,3,15,6,2,2,-1.,15,6,1,1,2.,16,7,1,1,2.,6.8655109498649836e-004,-0.0754187181591988,0.2640590965747833,0,2,3,1,6,3,-1.,5,1,2,3,3.,-0.0167514607310295,-0.6772903203964233,0.0329187288880348,0,3,7,16,2,2,-1.,7,16,1,1,2.,8,17,1,1,2.,-2.6301678735762835e-004,0.2272586971521378,-0.0905348733067513,0,3,5,17,2,2,-1.,5,17,1,1,2.,6,18,1,1,2.,4.3398610432632267e-004,0.0558943785727024,-0.3559266924858093,0,2,9,10,6,10,-1.,11,10,2,10,3.,-0.0201501492410898,0.1916276067495346,-0.0949299708008766,0,2,10,17,6,3,-1.,12,17,2,3,3.,-0.0144521296024323,-0.6851034164428711,0.0254221707582474,0,2,14,5,2,10,-1.,14,10,2,5,2.,-0.0211497396230698,0.3753319084644318,-0.0514965802431107,0,2,11,12,6,2,-1.,11,13,6,1,2.,0.0211377702653408,0.0290830805897713,-0.8943036794662476,0,2,8,1,1,3,-1.,8,2,1,1,3.,1.1524349683895707e-003,-0.0696949362754822,0.2729980051517487,0,3,12,15,2,2,-1.,12,15,1,1,2.,13,16,1,1,2.,-1.9070580310653895e-004,0.1822811961174011,-0.0983670726418495,0,3,6,8,6,4,-1.,6,8,3,2,2.,9,10,3,2,2.,-0.0363496318459511,-0.8369309902191162,0.0250557605177164,0,2,7,5,3,5,-1.,8,5,1,5,3.,-9.0632075443863869e-003,0.4146350026130676,-0.0544134490191936,0,2,0,5,7,3,-1.,0,6,7,1,3.,-2.0535490475594997e-003,-0.1975031048059464,0.1050689965486527,-0.9774749279022217,93,0,2,7,9,6,6,-1.,9,9,2,6,3.,-0.0227170195430517,0.2428855001926422,-0.1474552005529404,0,2,5,7,8,8,-1.,5,11,8,4,2.,0.0255059506744146,-0.2855173945426941,0.1083720996975899,0,3,4,9,2,6,-1.,4,9,1,3,2.,5,12,1,3,2.,-2.6640091091394424e-003,0.2927573025226593,-0.1037271022796631,0,2,10,11,6,1,-1.,12,11,2,1,3.,-3.8115289062261581e-003,0.2142689973115921,-0.1381113976240158,0,2,13,6,6,11,-1.,15,6,2,11,3.,-0.0167326908558607,0.2655026018619537,-0.0439113304018974,0,3,8,17,2,2,-1.,8,17,1,1,2.,9,18,1,1,2.,4.9277010839432478e-004,0.0211045593023300,-0.4297136068344116,0,2,4,12,12,1,-1.,8,12,4,1,3.,-0.0366911105811596,0.5399242043495178,-0.0436488017439842,0,2,11,17,3,2,-1.,11,18,3,1,2.,1.2615970335900784e-003,-0.1293386965990067,0.1663877069950104,0,2,8,17,6,1,-1.,10,17,2,1,3.,-8.4106856957077980e-003,-0.9469841122627258,0.0214658491313457,0,2,4,1,14,6,-1.,4,3,14,2,3.,0.0649027228355408,-0.0717277601361275,0.2661347985267639,0,2,14,2,2,12,-1.,14,8,2,6,2.,0.0303050000220537,-0.0827824920415878,0.2769432067871094,0,2,12,13,3,2,-1.,12,14,3,1,2.,2.5875340215861797e-003,-0.1296616941690445,0.1775663048028946,0,2,6,1,6,1,-1.,8,1,2,1,3.,-7.0240451022982597e-003,-0.6424317955970764,0.0399432107806206,0,2,10,6,6,1,-1.,12,6,2,1,3.,-1.0099769569933414e-003,0.1417661011219025,-0.1165997013449669,0,2,3,19,2,1,-1.,4,19,1,1,2.,-4.1179071558872238e-005,0.1568766981363297,-0.1112734004855156,0,3,18,16,2,2,-1.,18,16,1,1,2.,19,17,1,1,2.,-4.7293151146732271e-004,-0.3355455994606018,0.0459777303040028,0,2,16,11,3,7,-1.,17,11,1,7,3.,-1.7178079579025507e-003,0.1695290952920914,-0.1057806983590126,0,2,19,5,1,6,-1.,19,8,1,3,2.,-0.0133331697434187,-0.5825781226158142,0.0309784300625324,0,2,9,8,4,3,-1.,9,9,4,1,3.,-1.8783430568873882e-003,0.1426687985658646,-0.1113125979900360,0,3,16,8,4,4,-1.,16,8,2,2,2.,18,10,2,2,2.,-6.5765981562435627e-003,0.2756136059761047,-0.0531003288924694,0,3,2,8,2,2,-1.,2,8,1,1,2.,3,9,1,1,2.,-7.7210381277836859e-005,0.1324024051427841,-0.1116779968142510,0,3,3,5,6,4,-1.,3,5,3,2,2.,6,7,3,2,2.,0.0219685398042202,-0.0269681606441736,0.5006716847419739,0,3,2,3,8,16,-1.,2,3,4,8,2.,6,11,4,8,2.,-0.0274457503110170,-0.2408674061298370,0.0604782700538635,0,2,17,17,1,3,-1.,17,18,1,1,3.,7.8305849456228316e-005,-0.1333488970994949,0.1012346968054771,0,2,7,2,8,11,-1.,11,2,4,11,2.,0.0701906830072403,-0.0548637807369232,0.2480994015932083,0,2,13,3,6,14,-1.,16,3,3,14,2.,-0.0719021335244179,-0.3784669041633606,0.0422109998762608,0,2,0,9,18,2,-1.,6,9,6,2,3.,-0.1078097969293594,-0.3748658895492554,0.0428334400057793,0,2,6,10,14,3,-1.,6,11,14,1,3.,1.4364200178533792e-003,0.0804763585329056,-0.1726378947496414,0,2,10,9,9,3,-1.,13,9,3,3,3.,0.0682891905307770,-0.0355957895517349,0.4076131880283356,0,3,3,5,4,6,-1.,3,5,2,3,2.,5,8,2,3,2.,-6.8037179298698902e-003,0.1923379004001617,-0.0823680236935616,0,2,3,7,3,7,-1.,4,7,1,7,3.,-5.6193489581346512e-004,0.1305712014436722,-0.1435514986515045,0,2,2,8,11,6,-1.,2,10,11,2,3.,-0.0582766495645046,-0.3012543916702271,0.0528196506202221,0,2,8,9,6,3,-1.,8,10,6,1,3.,-6.1205718666315079e-003,0.2204390019178391,-0.0756917521357536,0,2,3,3,3,11,-1.,4,3,1,11,3.,-0.0135943097993732,-0.3904936015605927,0.0418571084737778,0,2,0,19,6,1,-1.,3,19,3,1,2.,1.3626200379803777e-003,-0.0953634232282639,0.1497032046318054,0,2,18,18,1,2,-1.,18,19,1,1,2.,-1.5074219845701009e-004,-0.2394558042287827,0.0647983327507973,0,3,8,0,12,6,-1.,8,0,6,3,2.,14,3,6,3,2.,-0.0774142593145370,0.5594198107719421,-0.0245168805122375,0,2,19,5,1,3,-1.,19,6,1,1,3.,9.2117872554808855e-004,0.0549288615584373,-0.2793481051921845,0,2,5,8,2,1,-1.,6,8,1,1,2.,1.0250780032947659e-003,-0.0621673092246056,0.2497636973857880,0,2,13,11,2,1,-1.,14,11,1,1,2.,-8.1174750812351704e-004,0.2343793958425522,-0.0657258108258247,0,2,3,6,15,13,-1.,8,6,5,13,3.,0.0834310203790665,0.0509548000991344,-0.3102098107337952,0,2,4,3,6,2,-1.,6,3,2,2,3.,-9.2014456167817116e-003,-0.3924253880977631,0.0329269506037235,0,2,0,18,1,2,-1.,0,19,1,1,2.,-2.9086650465615094e-004,-0.3103975057601929,0.0497118197381496,0,2,7,8,2,6,-1.,8,8,1,6,2.,7.7576898038387299e-003,-0.0440407507121563,0.3643135130405426,0,2,3,0,6,19,-1.,5,0,2,19,3.,-0.1246609017252922,-0.8195707798004150,0.0191506408154964,0,2,3,1,6,5,-1.,5,1,2,5,3.,0.0132425501942635,0.0389888398349285,-0.3323068022727966,0,2,17,14,3,6,-1.,17,16,3,2,3.,-6.6770128905773163e-003,-0.3579013943672180,0.0404602102935314,0,2,17,13,2,6,-1.,18,13,1,6,2.,-2.7479929849505424e-003,0.2525390088558197,-0.0564278215169907,0,2,17,18,2,2,-1.,18,18,1,2,2.,8.2659651525318623e-004,-0.0719886571168900,0.2278047949075699,0,2,11,14,9,4,-1.,14,14,3,4,3.,-0.0501534007489681,-0.6303647160530090,0.0274620503187180,0,3,15,8,4,6,-1.,15,8,2,3,2.,17,11,2,3,2.,7.4203149415552616e-003,-0.0666107162833214,0.2778733968734741,0,2,1,16,1,3,-1.,1,17,1,1,3.,-6.7951780511066318e-004,-0.3632706105709076,0.0427954308688641,0,2,7,0,3,14,-1.,8,0,1,14,3.,-1.9305750029161572e-003,0.1419623047113419,-0.1075998023152351,0,2,12,0,2,1,-1.,13,0,1,1,2.,-3.8132671033963561e-004,0.2159176021814346,-0.0702026635408401,0,2,7,9,6,5,-1.,10,9,3,5,2.,-0.0709903463721275,0.4526660144329071,-0.0407504811882973,0,2,15,5,4,9,-1.,17,5,2,9,2.,-0.0533680804073811,-0.6767405867576599,0.0192883405834436,0,2,11,0,6,6,-1.,13,0,2,6,3.,-0.0200648494064808,-0.4336543083190918,0.0318532884120941,0,3,16,15,2,2,-1.,16,15,1,1,2.,17,16,1,1,2.,1.1976360110566020e-003,-0.0265598706901073,0.5079718232154846,0,3,16,15,2,2,-1.,16,15,1,1,2.,17,16,1,1,2.,-2.2697300300933421e-004,0.1801259964704514,-0.0836065486073494,0,2,13,2,2,18,-1.,13,11,2,9,2.,0.0152626996859908,-0.2023892998695374,0.0674220174551010,0,2,8,4,8,10,-1.,8,9,8,5,2.,-0.2081176936626434,0.6694386005401611,-0.0224521104246378,0,2,8,3,2,3,-1.,8,4,2,1,3.,1.5514369588345289e-003,-0.0751218423247337,0.1732691973447800,0,2,11,1,6,9,-1.,11,4,6,3,3.,-0.0529240109026432,0.2499251961708069,-0.0628791674971581,0,2,15,4,5,6,-1.,15,6,5,2,3.,-0.0216488502919674,-0.2919428050518036,0.0526144914329052,0,3,12,18,2,2,-1.,12,18,1,1,2.,13,19,1,1,2.,-2.2905069636180997e-004,-0.2211730033159256,0.0631683394312859,0,2,1,17,1,3,-1.,1,18,1,1,3.,5.0170070608146489e-005,-0.1151070967316628,0.1161144003272057,0,2,12,19,2,1,-1.,13,19,1,1,2.,-1.6416069411206990e-004,0.1587152034044266,-0.0826006010174751,0,2,8,10,6,6,-1.,10,10,2,6,3.,-0.0120032895356417,0.1221809014678001,-0.1122969985008240,0,2,14,2,6,5,-1.,16,2,2,5,3.,-0.0177841000258923,-0.3507278859615326,0.0313419215381145,0,2,9,5,2,6,-1.,9,7,2,2,3.,-6.3457582145929337e-003,0.1307806968688965,-0.1057441011071205,0,2,1,15,2,2,-1.,2,15,1,2,2.,-7.9523242311552167e-004,0.1720467060804367,-0.0860019922256470,0,2,18,17,1,3,-1.,18,18,1,1,3.,-3.1029590172693133e-004,-0.2843317091464996,0.0518171191215515,0,2,10,14,4,6,-1.,10,16,4,2,3.,-0.0170537102967501,0.3924242854118347,-0.0401432700455189,0,2,9,7,3,2,-1.,10,7,1,2,3.,4.6504959464073181e-003,-0.0318375602364540,0.4123769998550415,0,3,6,9,6,2,-1.,6,9,3,1,2.,9,10,3,1,2.,-0.0103587601333857,-0.5699319839477539,0.0292483791708946,0,2,0,2,1,12,-1.,0,6,1,4,3.,-0.0221962407231331,-0.4560528993606567,0.0262859892100096,0,2,4,0,15,1,-1.,9,0,5,1,3.,-7.0536029525101185e-003,0.1599832028150559,-0.0915948599576950,0,3,9,0,8,2,-1.,9,0,4,1,2.,13,1,4,1,2.,-5.7094299700111151e-004,-0.1407632976770401,0.1028741970658302,0,2,12,2,8,1,-1.,16,2,4,1,2.,-2.2152599412947893e-003,0.1659359931945801,-0.0852739885449409,0,2,7,1,10,6,-1.,7,3,10,2,3.,-0.0280848909169436,0.2702234089374542,-0.0558738112449646,0,2,18,6,2,3,-1.,18,7,2,1,3.,2.1515151020139456e-003,0.0424728915095329,-0.3200584948062897,0,3,4,12,2,2,-1.,4,12,1,1,2.,5,13,1,1,2.,-2.9733829433098435e-004,0.1617716997861862,-0.0851155892014503,0,2,6,6,6,2,-1.,8,6,2,2,3.,-0.0166947804391384,-0.4285877048969269,0.0305416099727154,0,2,0,9,9,6,-1.,3,9,3,6,3.,0.1198299005627632,-0.0162772908806801,0.7984678149223328,0,2,17,18,2,2,-1.,18,18,1,2,2.,-3.5499420482665300e-004,0.1593593955039978,-0.0832728818058968,0,2,11,2,6,16,-1.,13,2,2,16,3.,-0.0182262696325779,0.1952728033065796,-0.0739398896694183,0,2,2,4,15,13,-1.,7,4,5,13,3.,-4.0238600922748446e-004,0.0791018083691597,-0.2080612927675247,0,2,16,2,3,10,-1.,17,2,1,10,3.,4.0892060496844351e-004,0.1003663018345833,-0.1512821018695831,0,2,6,10,2,1,-1.,7,10,1,1,2.,9.5368112670257688e-004,-0.0730116665363312,0.2175202071666718,0,2,1,1,18,16,-1.,10,1,9,16,2.,0.4308179914951325,-0.0274506993591785,0.5706158280372620,0,2,14,4,3,15,-1.,15,4,1,15,3.,5.3564831614494324e-004,0.1158754006028175,-0.1279056072235107,0,2,19,13,1,2,-1.,19,14,1,1,2.,2.4430730263702571e-005,-0.1681662946939468,0.0804499834775925,0,2,2,6,5,8,-1.,2,10,5,4,2.,-0.0553456507623196,0.4533894956111908,-0.0312227793037891];
    module.eye = new Float32Array(classifier);
    module.eye.tilted = false;
})(objectdetect);
(function(module) {
	"use strict";
	
    var classifier = [20,20,0.8226894140243530,3,0,2,3,7,14,4,-1.,3,9,14,2,2.,4.0141958743333817e-003,0.0337941907346249,0.8378106951713562,0,2,1,2,18,4,-1.,7,2,6,4,3.,0.0151513395830989,0.1514132022857666,0.7488812208175659,0,2,1,7,15,9,-1.,1,10,15,3,3.,4.2109931819140911e-003,0.0900492817163467,0.6374819874763489,6.9566087722778320,16,0,2,5,6,2,6,-1.,5,9,2,3,2.,1.6227109590545297e-003,0.0693085864186287,0.7110946178436279,0,2,7,5,6,3,-1.,9,5,2,3,3.,2.2906649392098188e-003,0.1795803010463715,0.6668692231178284,0,2,4,0,12,9,-1.,4,3,12,3,3.,5.0025708042085171e-003,0.1693672984838486,0.6554006934165955,0,2,6,9,10,8,-1.,6,13,10,4,2.,7.9659894108772278e-003,0.5866332054138184,0.0914145186543465,0,2,3,6,14,8,-1.,3,10,14,4,2.,-3.5227010957896709e-003,0.1413166970014572,0.6031895875930786,0,2,14,1,6,10,-1.,14,1,3,10,2.,0.0366676896810532,0.3675672113895416,0.7920318245887756,0,2,7,8,5,12,-1.,7,12,5,4,3.,9.3361474573612213e-003,0.6161385774612427,0.2088509947061539,0,2,1,1,18,3,-1.,7,1,6,3,3.,8.6961314082145691e-003,0.2836230993270874,0.6360273957252502,0,2,1,8,17,2,-1.,1,9,17,1,2.,1.1488880263641477e-003,0.2223580926656723,0.5800700783729553,0,2,16,6,4,2,-1.,16,7,4,1,2.,-2.1484689787030220e-003,0.2406464070081711,0.5787054896354675,0,2,5,17,2,2,-1.,5,18,2,1,2.,2.1219060290604830e-003,0.5559654831886292,0.1362237036228180,0,2,14,2,6,12,-1.,14,2,3,12,2.,-0.0939491465687752,0.8502737283706665,0.4717740118503571,0,3,4,0,4,12,-1.,4,0,2,6,2.,6,6,2,6,2.,1.3777789426967502e-003,0.5993673801422119,0.2834529876708984,0,2,2,11,18,8,-1.,8,11,6,8,3.,0.0730631574988365,0.4341886043548584,0.7060034275054932,0,2,5,7,10,2,-1.,5,8,10,1,2.,3.6767389974556863e-004,0.3027887940406799,0.6051574945449829,0,2,15,11,5,3,-1.,15,12,5,1,3.,-6.0479710809886456e-003,0.1798433959484100,0.5675256848335266,9.4985427856445313,21,0,2,5,3,10,9,-1.,5,6,10,3,3.,-0.0165106896311045,0.6644225120544434,0.1424857974052429,0,2,9,4,2,14,-1.,9,11,2,7,2.,2.7052499353885651e-003,0.6325352191925049,0.1288477033376694,0,2,3,5,4,12,-1.,3,9,4,4,3.,2.8069869149476290e-003,0.1240288019180298,0.6193193197250366,0,2,4,5,12,5,-1.,8,5,4,5,3.,-1.5402400167658925e-003,0.1432143002748489,0.5670015811920166,0,2,5,6,10,8,-1.,5,10,10,4,2.,-5.6386279175058007e-004,0.1657433062791824,0.5905207991600037,0,2,8,0,6,9,-1.,8,3,6,3,3.,1.9253729842603207e-003,0.2695507109165192,0.5738824009895325,0,2,9,12,1,8,-1.,9,16,1,4,2.,-5.0214841030538082e-003,0.1893538981676102,0.5782774090766907,0,2,0,7,20,6,-1.,0,9,20,2,3.,2.6365420781075954e-003,0.2309329062700272,0.5695425868034363,0,2,7,0,6,17,-1.,9,0,2,17,3.,-1.5127769438549876e-003,0.2759602069854736,0.5956642031669617,0,2,9,0,6,4,-1.,11,0,2,4,3.,-0.0101574398577213,0.1732538044452667,0.5522047281265259,0,2,5,1,6,4,-1.,7,1,2,4,3.,-0.0119536602869630,0.1339409947395325,0.5559014081954956,0,2,12,1,6,16,-1.,14,1,2,16,3.,4.8859491944313049e-003,0.3628703951835632,0.6188849210739136,0,3,0,5,18,8,-1.,0,5,9,4,2.,9,9,9,4,2.,-0.0801329165697098,0.0912110507488251,0.5475944876670837,0,3,8,15,10,4,-1.,13,15,5,2,2.,8,17,5,2,2.,1.0643280111253262e-003,0.3715142905712128,0.5711399912834168,0,3,3,1,4,8,-1.,3,1,2,4,2.,5,5,2,4,2.,-1.3419450260698795e-003,0.5953313708305359,0.3318097889423370,0,3,3,6,14,10,-1.,10,6,7,5,2.,3,11,7,5,2.,-0.0546011403203011,0.1844065934419632,0.5602846145629883,0,2,2,1,6,16,-1.,4,1,2,16,3.,2.9071690514683723e-003,0.3594244122505188,0.6131715178489685,0,2,0,18,20,2,-1.,0,19,20,1,2.,7.4718717951327562e-004,0.5994353294372559,0.3459562957286835,0,2,8,13,4,3,-1.,8,14,4,1,3.,4.3013808317482471e-003,0.4172652065753937,0.6990845203399658,0,2,9,14,2,3,-1.,9,15,2,1,3.,4.5017572119832039e-003,0.4509715139865875,0.7801457047462463,0,2,0,12,9,6,-1.,0,14,9,2,3.,0.0241385009139776,0.5438212752342224,0.1319826990365982,18.4129695892333980,39,0,2,5,7,3,4,-1.,5,9,3,2,2.,1.9212230108678341e-003,0.1415266990661621,0.6199870705604553,0,2,9,3,2,16,-1.,9,11,2,8,2.,-1.2748669541906565e-004,0.6191074252128601,0.1884928941726685,0,2,3,6,13,8,-1.,3,10,13,4,2.,5.1409931620582938e-004,0.1487396955490112,0.5857927799224854,0,2,12,3,8,2,-1.,12,3,4,2,2.,4.1878609918057919e-003,0.2746909856796265,0.6359239816665649,0,2,8,8,4,12,-1.,8,12,4,4,3.,5.1015717908740044e-003,0.5870851278305054,0.2175628989934921,0,3,11,3,8,6,-1.,15,3,4,3,2.,11,6,4,3,2.,-2.1448440384119749e-003,0.5880944728851318,0.2979590892791748,0,2,7,1,6,19,-1.,9,1,2,19,3.,-2.8977119363844395e-003,0.2373327016830444,0.5876647233963013,0,2,9,0,6,4,-1.,11,0,2,4,3.,-0.0216106791049242,0.1220654994249344,0.5194202065467835,0,2,3,1,9,3,-1.,6,1,3,3,3.,-4.6299318782985210e-003,0.2631230950355530,0.5817409157752991,0,3,8,15,10,4,-1.,13,15,5,2,2.,8,17,5,2,2.,5.9393711853772402e-004,0.3638620078563690,0.5698544979095459,0,2,0,3,6,10,-1.,3,3,3,10,2.,0.0538786612451077,0.4303531050682068,0.7559366226196289,0,2,3,4,15,15,-1.,3,9,15,5,3.,1.8887349870055914e-003,0.2122603058815002,0.5613427162170410,0,2,6,5,8,6,-1.,6,7,8,2,3.,-2.3635339457541704e-003,0.5631849169731140,0.2642767131328583,0,3,4,4,12,10,-1.,10,4,6,5,2.,4,9,6,5,2.,0.0240177996456623,0.5797107815742493,0.2751705944538117,0,2,6,4,4,4,-1.,8,4,2,4,2.,2.0543030404951423e-004,0.2705242037773132,0.5752568840980530,0,2,15,11,1,2,-1.,15,12,1,1,2.,8.4790197433903813e-004,0.5435624718666077,0.2334876954555512,0,2,3,11,2,2,-1.,3,12,2,1,2.,1.4091329649090767e-003,0.5319424867630005,0.2063155025243759,0,2,16,11,1,3,-1.,16,12,1,1,3.,1.4642629539594054e-003,0.5418980717658997,0.3068861067295075,0,3,3,15,6,4,-1.,3,15,3,2,2.,6,17,3,2,2.,1.6352549428120255e-003,0.3695372939109802,0.6112868189811707,0,2,6,7,8,2,-1.,6,8,8,1,2.,8.3172752056270838e-004,0.3565036952495575,0.6025236248970032,0,2,3,11,1,3,-1.,3,12,1,1,3.,-2.0998890977352858e-003,0.1913982033729553,0.5362827181816101,0,2,6,0,12,2,-1.,6,1,12,1,2.,-7.4213981861248612e-004,0.3835555016994476,0.5529310107231140,0,2,9,14,2,3,-1.,9,15,2,1,3.,3.2655049581080675e-003,0.4312896132469177,0.7101895809173584,0,2,7,15,6,2,-1.,7,16,6,1,2.,8.9134991867467761e-004,0.3984830975532532,0.6391963958740234,0,2,0,5,4,6,-1.,0,7,4,2,3.,-0.0152841797098517,0.2366732954978943,0.5433713793754578,0,2,4,12,12,2,-1.,8,12,4,2,3.,4.8381411470472813e-003,0.5817500948905945,0.3239189088344574,0,2,6,3,1,9,-1.,6,6,1,3,3.,-9.1093179071322083e-004,0.5540593862533569,0.2911868989467621,0,2,10,17,3,2,-1.,11,17,1,2,3.,-6.1275060288608074e-003,0.1775255054235458,0.5196629166603088,0,2,9,9,2,2,-1.,9,10,2,1,2.,-4.4576259097084403e-004,0.3024170100688934,0.5533593893051148,0,2,7,6,6,4,-1.,9,6,2,4,3.,0.0226465407758951,0.4414930939674377,0.6975377202033997,0,2,7,17,3,2,-1.,8,17,1,2,3.,-1.8804960418492556e-003,0.2791394889354706,0.5497952103614807,0,2,10,17,3,3,-1.,11,17,1,3,3.,7.0889107882976532e-003,0.5263199210166931,0.2385547012090683,0,2,8,12,3,2,-1.,8,13,3,1,2.,1.7318050377070904e-003,0.4319379031658173,0.6983600854873657,0,2,9,3,6,2,-1.,11,3,2,2,3.,-6.8482700735330582e-003,0.3082042932510376,0.5390920042991638,0,2,3,11,14,4,-1.,3,13,14,2,2.,-1.5062530110299122e-005,0.5521922111511231,0.3120366036891937,0,3,1,10,18,4,-1.,10,10,9,2,2.,1,12,9,2,2.,0.0294755697250366,0.5401322841644287,0.1770603060722351,0,2,0,10,3,3,-1.,0,11,3,1,3.,8.1387329846620560e-003,0.5178617835044861,0.1211019009351730,0,2,9,1,6,6,-1.,11,1,2,6,3.,0.0209429506212473,0.5290294289588928,0.3311221897602081,0,2,8,7,3,6,-1.,9,7,1,6,3.,-9.5665529370307922e-003,0.7471994161605835,0.4451968967914581,15.3241395950317380,33,0,2,1,0,18,9,-1.,1,3,18,3,3.,-2.8206960996612906e-004,0.2064086049795151,0.6076732277870178,0,2,12,10,2,6,-1.,12,13,2,3,2.,1.6790600493550301e-003,0.5851997137069702,0.1255383938550949,0,2,0,5,19,8,-1.,0,9,19,4,2.,6.9827912375330925e-004,0.0940184295177460,0.5728961229324341,0,2,7,0,6,9,-1.,9,0,2,9,3.,7.8959012171253562e-004,0.1781987994909287,0.5694308876991272,0,2,5,3,6,1,-1.,7,3,2,1,3.,-2.8560499195009470e-003,0.1638399064540863,0.5788664817810059,0,2,11,3,6,1,-1.,13,3,2,1,3.,-3.8122469559311867e-003,0.2085440009832382,0.5508564710617065,0,2,5,10,4,6,-1.,5,13,4,3,2.,1.5896620461717248e-003,0.5702760815620422,0.1857215017080307,0,2,11,3,6,1,-1.,13,3,2,1,3.,0.0100783398374915,0.5116943120956421,0.2189770042896271,0,2,4,4,12,6,-1.,4,6,12,2,3.,-0.0635263025760651,0.7131379842758179,0.4043813049793243,0,2,15,12,2,6,-1.,15,14,2,2,3.,-9.1031491756439209e-003,0.2567181885242462,0.5463973283767700,0,2,9,3,2,2,-1.,10,3,1,2,2.,-2.4035000242292881e-003,0.1700665950775147,0.5590974092483521,0,2,9,3,3,1,-1.,10,3,1,1,3.,1.5226360410451889e-003,0.5410556793212891,0.2619054019451141,0,2,1,1,4,14,-1.,3,1,2,14,2.,0.0179974399507046,0.3732436895370483,0.6535220742225647,0,3,9,0,4,4,-1.,11,0,2,2,2.,9,2,2,2,2.,-6.4538191072642803e-003,0.2626481950283051,0.5537446141242981,0,2,7,5,1,14,-1.,7,12,1,7,2.,-0.0118807600811124,0.2003753930330277,0.5544745922088623,0,2,19,0,1,4,-1.,19,2,1,2,2.,1.2713660253211856e-003,0.5591902732849121,0.3031975924968720,0,2,5,5,6,4,-1.,8,5,3,4,2.,1.1376109905540943e-003,0.2730407118797302,0.5646508932113648,0,2,9,18,3,2,-1.,10,18,1,2,3.,-4.2651998810470104e-003,0.1405909061431885,0.5461820960044861,0,2,8,18,3,2,-1.,9,18,1,2,3.,-2.9602861031889915e-003,0.1795035004615784,0.5459290146827698,0,2,4,5,12,6,-1.,4,7,12,2,3.,-8.8448226451873779e-003,0.5736783146858215,0.2809219956398010,0,2,3,12,2,6,-1.,3,14,2,2,3.,-6.6430689767003059e-003,0.2370675951242447,0.5503826141357422,0,2,10,8,2,12,-1.,10,12,2,4,3.,3.9997808635234833e-003,0.5608199834823608,0.3304282128810883,0,2,7,18,3,2,-1.,8,18,1,2,3.,-4.1221720166504383e-003,0.1640105992555618,0.5378993153572083,0,2,9,0,6,2,-1.,11,0,2,2,3.,0.0156249096617103,0.5227649211883545,0.2288603931665421,0,2,5,11,9,3,-1.,5,12,9,1,3.,-0.0103564197197557,0.7016193866729736,0.4252927899360657,0,2,9,0,6,2,-1.,11,0,2,2,3.,-8.7960809469223022e-003,0.2767347097396851,0.5355830192565918,0,2,1,1,18,5,-1.,7,1,6,5,3.,0.1622693985700607,0.4342240095138550,0.7442579269409180,0,3,8,0,4,4,-1.,10,0,2,2,2.,8,2,2,2,2.,4.5542530715465546e-003,0.5726485848426819,0.2582125067710877,0,2,3,12,1,3,-1.,3,13,1,1,3.,-2.1309209987521172e-003,0.2106848061084747,0.5361018776893616,0,2,8,14,5,3,-1.,8,15,5,1,3.,-0.0132084200158715,0.7593790888786316,0.4552468061447144,0,3,5,4,10,12,-1.,5,4,5,6,2.,10,10,5,6,2.,-0.0659966766834259,0.1252475976943970,0.5344039797782898,0,2,9,6,9,12,-1.,9,10,9,4,3.,7.9142656177282333e-003,0.3315384089946747,0.5601043105125427,0,3,2,2,12,14,-1.,2,2,6,7,2.,8,9,6,7,2.,0.0208942797034979,0.5506049990653992,0.2768838107585907,21.0106391906738280,44,0,2,4,7,12,2,-1.,8,7,4,2,3.,1.1961159761995077e-003,0.1762690991163254,0.6156241297721863,0,2,7,4,6,4,-1.,7,6,6,2,2.,-1.8679830245673656e-003,0.6118106842041016,0.1832399964332581,0,2,4,5,11,8,-1.,4,9,11,4,2.,-1.9579799845814705e-004,0.0990442633628845,0.5723816156387329,0,2,3,10,16,4,-1.,3,12,16,2,2.,-8.0255657667294145e-004,0.5579879879951477,0.2377282977104187,0,2,0,0,16,2,-1.,0,1,16,1,2.,-2.4510810617357492e-003,0.2231457978487015,0.5858935117721558,0,2,7,5,6,2,-1.,9,5,2,2,3.,5.0361850298941135e-004,0.2653993964195252,0.5794103741645813,0,3,3,2,6,10,-1.,3,2,3,5,2.,6,7,3,5,2.,4.0293349884450436e-003,0.5803827047348023,0.2484865039587021,0,2,10,5,8,15,-1.,10,10,8,5,3.,-0.0144517095759511,0.1830351948738098,0.5484204888343811,0,3,3,14,8,6,-1.,3,14,4,3,2.,7,17,4,3,2.,2.0380979403853416e-003,0.3363558948040009,0.6051092743873596,0,2,14,2,2,2,-1.,14,3,2,1,2.,-1.6155190533027053e-003,0.2286642044782639,0.5441246032714844,0,2,1,10,7,6,-1.,1,13,7,3,2.,3.3458340913057327e-003,0.5625913143157959,0.2392338067293167,0,2,15,4,4,3,-1.,15,4,2,3,2.,1.6379579901695251e-003,0.3906993865966797,0.5964621901512146,0,3,2,9,14,6,-1.,2,9,7,3,2.,9,12,7,3,2.,0.0302512105554342,0.5248482227325440,0.1575746983289719,0,2,5,7,10,4,-1.,5,9,10,2,2.,0.0372519902884960,0.4194310903549194,0.6748418807983398,0,3,6,9,8,8,-1.,6,9,4,4,2.,10,13,4,4,2.,-0.0251097902655602,0.1882549971342087,0.5473451018333435,0,2,14,1,3,2,-1.,14,2,3,1,2.,-5.3099058568477631e-003,0.1339973062276840,0.5227110981941223,0,2,1,4,4,2,-1.,3,4,2,2,2.,1.2086479691788554e-003,0.3762088119983673,0.6109635829925537,0,2,11,10,2,8,-1.,11,14,2,4,2.,-0.0219076797366142,0.2663142979145050,0.5404006838798523,0,2,0,0,5,3,-1.,0,1,5,1,3.,5.4116579703986645e-003,0.5363578796386719,0.2232273072004318,0,3,2,5,18,8,-1.,11,5,9,4,2.,2,9,9,4,2.,0.0699463263154030,0.5358232855796814,0.2453698068857193,0,2,6,6,1,6,-1.,6,9,1,3,2.,3.4520021290518343e-004,0.2409671992063522,0.5376930236816406,0,2,19,1,1,3,-1.,19,2,1,1,3.,1.2627709656953812e-003,0.5425856709480286,0.3155693113803864,0,2,7,6,6,6,-1.,9,6,2,6,3.,0.0227195098996162,0.4158405959606171,0.6597865223884583,0,2,19,1,1,3,-1.,19,2,1,1,3.,-1.8111000536009669e-003,0.2811253070831299,0.5505244731903076,0,2,3,13,2,3,-1.,3,14,2,1,3.,3.3469670452177525e-003,0.5260028243064880,0.1891465038061142,0,3,8,4,8,12,-1.,12,4,4,6,2.,8,10,4,6,2.,4.0791751234792173e-004,0.5673509240150452,0.3344210088253021,0,2,5,2,6,3,-1.,7,2,2,3,3.,0.0127347996458411,0.5343592166900635,0.2395612001419067,0,2,6,1,9,10,-1.,6,6,9,5,2.,-7.3119727894663811e-003,0.6010890007019043,0.4022207856178284,0,2,0,4,6,12,-1.,2,4,2,12,3.,-0.0569487512111664,0.8199151158332825,0.4543190896511078,0,2,15,13,2,3,-1.,15,14,2,1,3.,-5.0116591155529022e-003,0.2200281023979187,0.5357710719108582,0,2,7,14,5,3,-1.,7,15,5,1,3.,6.0334368608891964e-003,0.4413081109523773,0.7181751132011414,0,2,15,13,3,3,-1.,15,14,3,1,3.,3.9437441155314445e-003,0.5478860735893250,0.2791733145713806,0,2,6,14,8,3,-1.,6,15,8,1,3.,-3.6591119132936001e-003,0.6357867717742920,0.3989723920822144,0,2,15,13,3,3,-1.,15,14,3,1,3.,-3.8456181064248085e-003,0.3493686020374298,0.5300664901733398,0,2,2,13,3,3,-1.,2,14,3,1,3.,-7.1926261298358440e-003,0.1119614988565445,0.5229672789573669,0,3,4,7,12,12,-1.,10,7,6,6,2.,4,13,6,6,2.,-0.0527989417314529,0.2387102991342545,0.5453451275825501,0,2,9,7,2,6,-1.,10,7,1,6,2.,-7.9537667334079742e-003,0.7586917877197266,0.4439376890659332,0,2,8,9,5,2,-1.,8,10,5,1,2.,-2.7344180271029472e-003,0.2565476894378662,0.5489321947097778,0,2,8,6,3,4,-1.,9,6,1,4,3.,-1.8507939530536532e-003,0.6734347939491272,0.4252474904060364,0,2,9,6,2,8,-1.,9,10,2,4,2.,0.0159189198166132,0.5488352775573731,0.2292661964893341,0,2,7,7,3,6,-1.,8,7,1,6,3.,-1.2687679845839739e-003,0.6104331016540527,0.4022389948368073,0,2,11,3,3,3,-1.,12,3,1,3,3.,6.2883910723030567e-003,0.5310853123664856,0.1536193042993546,0,2,5,4,6,1,-1.,7,4,2,1,3.,-6.2259892001748085e-003,0.1729111969470978,0.5241606235504150,0,2,5,6,10,3,-1.,5,7,10,1,3.,-0.0121325999498367,0.6597759723663330,0.4325182139873505,23.9187908172607420,50,0,2,7,3,6,9,-1.,7,6,6,3,3.,-3.9184908382594585e-003,0.6103435158729553,0.1469330936670303,0,2,6,7,9,1,-1.,9,7,3,1,3.,1.5971299726516008e-003,0.2632363140583038,0.5896466970443726,0,2,2,8,16,8,-1.,2,12,16,4,2.,0.0177801102399826,0.5872874259948731,0.1760361939668655,0,2,14,6,2,6,-1.,14,9,2,3,2.,6.5334769897162914e-004,0.1567801982164383,0.5596066117286682,0,2,1,5,6,15,-1.,1,10,6,5,3.,-2.8353091329336166e-004,0.1913153976202011,0.5732036232948303,0,2,10,0,6,9,-1.,10,3,6,3,3.,1.6104689566418529e-003,0.2914913892745972,0.5623080730438232,0,2,6,6,7,14,-1.,6,13,7,7,2.,-0.0977506190538406,0.1943476945161820,0.5648233294487000,0,2,13,7,3,6,-1.,13,9,3,2,3.,5.5182358482852578e-004,0.3134616911411285,0.5504639744758606,0,2,1,8,15,4,-1.,6,8,5,4,3.,-0.0128582203760743,0.2536481916904450,0.5760142803192139,0,2,11,2,3,10,-1.,11,7,3,5,2.,4.1530239395797253e-003,0.5767722129821777,0.3659774065017700,0,2,3,7,4,6,-1.,3,9,4,2,3.,1.7092459602281451e-003,0.2843191027641296,0.5918939113616943,0,2,13,3,6,10,-1.,15,3,2,10,3.,7.5217359699308872e-003,0.4052427113056183,0.6183109283447266,0,3,5,7,8,10,-1.,5,7,4,5,2.,9,12,4,5,2.,2.2479810286313295e-003,0.5783755183219910,0.3135401010513306,0,3,4,4,12,12,-1.,10,4,6,6,2.,4,10,6,6,2.,0.0520062111318111,0.5541312098503113,0.1916636973619461,0,2,1,4,6,9,-1.,3,4,2,9,3.,0.0120855299755931,0.4032655954360962,0.6644591093063355,0,2,11,3,2,5,-1.,11,3,1,5,2.,1.4687820112158079e-005,0.3535977900028229,0.5709382891654968,0,2,7,3,2,5,-1.,8,3,1,5,2.,7.1395188570022583e-006,0.3037444949150085,0.5610269904136658,0,2,10,14,2,3,-1.,10,15,2,1,3.,-4.6001640148460865e-003,0.7181087136268616,0.4580326080322266,0,2,5,12,6,2,-1.,8,12,3,2,2.,2.0058949012309313e-003,0.5621951818466187,0.2953684031963348,0,2,9,14,2,3,-1.,9,15,2,1,3.,4.5050270855426788e-003,0.4615387916564941,0.7619017958641052,0,2,4,11,12,6,-1.,4,14,12,3,2.,0.0117468303069472,0.5343837141990662,0.1772529035806656,0,2,11,11,5,9,-1.,11,14,5,3,3.,-0.0583163388073444,0.1686245948076248,0.5340772271156311,0,2,6,15,3,2,-1.,6,16,3,1,2.,2.3629379575140774e-004,0.3792056143283844,0.6026803851127625,0,2,11,0,3,5,-1.,12,0,1,5,3.,-7.8156180679798126e-003,0.1512867063283920,0.5324323773384094,0,2,5,5,6,7,-1.,8,5,3,7,2.,-0.0108761601150036,0.2081822007894516,0.5319945216178894,0,2,13,0,1,9,-1.,13,3,1,3,3.,-2.7745519764721394e-003,0.4098246991634369,0.5210328102111816,0,3,3,2,4,8,-1.,3,2,2,4,2.,5,6,2,4,2.,-7.8276381827890873e-004,0.5693274140357971,0.3478842079639435,0,2,13,12,4,6,-1.,13,14,4,2,3.,0.0138704096898437,0.5326750874519348,0.2257698029279709,0,2,3,12,4,6,-1.,3,14,4,2,3.,-0.0236749108880758,0.1551305055618286,0.5200707912445068,0,2,13,11,3,4,-1.,13,13,3,2,2.,-1.4879409718560055e-005,0.5500566959381104,0.3820176124572754,0,2,4,4,4,3,-1.,4,5,4,1,3.,3.6190641112625599e-003,0.4238683879375458,0.6639748215675354,0,2,7,5,11,8,-1.,7,9,11,4,2.,-0.0198171101510525,0.2150038033723831,0.5382357835769653,0,2,7,8,3,4,-1.,8,8,1,4,3.,-3.8154039066284895e-003,0.6675711274147034,0.4215297102928162,0,2,9,1,6,1,-1.,11,1,2,1,3.,-4.9775829538702965e-003,0.2267289012670517,0.5386328101158142,0,2,5,5,3,3,-1.,5,6,3,1,3.,2.2441020701080561e-003,0.4308691024780273,0.6855735778808594,0,3,0,9,20,6,-1.,10,9,10,3,2.,0,12,10,3,2.,0.0122824599966407,0.5836614966392517,0.3467479050159454,0,2,8,6,3,5,-1.,9,6,1,5,3.,-2.8548699337989092e-003,0.7016944885253906,0.4311453998088837,0,2,11,0,1,3,-1.,11,1,1,1,3.,-3.7875669077038765e-003,0.2895345091819763,0.5224946141242981,0,2,4,2,4,2,-1.,4,3,4,1,2.,-1.2201230274513364e-003,0.2975570857524872,0.5481644868850708,0,2,12,6,4,3,-1.,12,7,4,1,3.,0.0101605998352170,0.4888817965984345,0.8182697892189026,0,2,5,0,6,4,-1.,7,0,2,4,3.,-0.0161745697259903,0.1481492966413498,0.5239992737770081,0,2,9,7,3,8,-1.,10,7,1,8,3.,0.0192924607545137,0.4786309897899628,0.7378190755844116,0,2,9,7,2,2,-1.,10,7,1,2,2.,-3.2479539513587952e-003,0.7374222874641419,0.4470643997192383,0,3,6,7,14,4,-1.,13,7,7,2,2.,6,9,7,2,2.,-9.3803480267524719e-003,0.3489154875278473,0.5537996292114258,0,2,0,5,3,6,-1.,0,7,3,2,3.,-0.0126061299815774,0.2379686981439591,0.5315443277359009,0,2,13,11,3,4,-1.,13,13,3,2,2.,-0.0256219301372766,0.1964688003063202,0.5138769745826721,0,2,4,11,3,4,-1.,4,13,3,2,2.,-7.5741496402770281e-005,0.5590522885322571,0.3365853130817413,0,3,5,9,12,8,-1.,11,9,6,4,2.,5,13,6,4,2.,-0.0892108827829361,0.0634046569466591,0.5162634849548340,0,2,9,12,1,3,-1.,9,13,1,1,3.,-2.7670480776578188e-003,0.7323467731475830,0.4490706026554108,0,2,10,15,2,4,-1.,10,17,2,2,2.,2.7152578695677221e-004,0.4114834964275360,0.5985518097877502,24.5278797149658200,51,0,2,7,7,6,1,-1.,9,7,2,1,3.,1.4786219689995050e-003,0.2663545012474060,0.6643316745758057,0,3,12,3,6,6,-1.,15,3,3,3,2.,12,6,3,3,2.,-1.8741659587249160e-003,0.6143848896026611,0.2518512904644013,0,2,0,4,10,6,-1.,0,6,10,2,3.,-1.7151009524241090e-003,0.5766341090202332,0.2397463023662567,0,3,8,3,8,14,-1.,12,3,4,7,2.,8,10,4,7,2.,-1.8939269939437509e-003,0.5682045817375183,0.2529144883155823,0,2,4,4,7,15,-1.,4,9,7,5,3.,-5.3006052039563656e-003,0.1640675961971283,0.5556079745292664,0,3,12,2,6,8,-1.,15,2,3,4,2.,12,6,3,4,2.,-0.0466625317931175,0.6123154163360596,0.4762830138206482,0,3,2,2,6,8,-1.,2,2,3,4,2.,5,6,3,4,2.,-7.9431332414969802e-004,0.5707858800888062,0.2839404046535492,0,2,2,13,18,7,-1.,8,13,6,7,3.,0.0148916700854898,0.4089672863483429,0.6006367206573486,0,3,4,3,8,14,-1.,4,3,4,7,2.,8,10,4,7,2.,-1.2046529445797205e-003,0.5712450742721558,0.2705289125442505,0,2,18,1,2,6,-1.,18,3,2,2,3.,6.0619381256401539e-003,0.5262504220008850,0.3262225985527039,0,2,9,11,2,3,-1.,9,12,2,1,3.,-2.5286648888140917e-003,0.6853830814361572,0.4199256896972656,0,2,18,1,2,6,-1.,18,3,2,2,3.,-5.9010218828916550e-003,0.3266282081604004,0.5434812903404236,0,2,0,1,2,6,-1.,0,3,2,2,3.,5.6702760048210621e-003,0.5468410849571228,0.2319003939628601,0,2,1,5,18,6,-1.,1,7,18,2,3.,-3.0304100364446640e-003,0.5570667982101440,0.2708238065242767,0,2,0,2,6,7,-1.,3,2,3,7,2.,2.9803649522364140e-003,0.3700568974018097,0.5890625715255737,0,2,7,3,6,14,-1.,7,10,6,7,2.,-0.0758405104279518,0.2140070050954819,0.5419948101043701,0,2,3,7,13,10,-1.,3,12,13,5,2.,0.0192625392228365,0.5526772141456604,0.2726590037345886,0,2,11,15,2,2,-1.,11,16,2,1,2.,1.8888259364757687e-004,0.3958011865615845,0.6017209887504578,0,3,2,11,16,4,-1.,2,11,8,2,2.,10,13,8,2,2.,0.0293695498257875,0.5241373777389526,0.1435758024454117,0,3,13,7,6,4,-1.,16,7,3,2,2.,13,9,3,2,2.,1.0417619487270713e-003,0.3385409116744995,0.5929983258247376,0,2,6,10,3,9,-1.,6,13,3,3,3.,2.6125640142709017e-003,0.5485377907752991,0.3021597862243652,0,2,14,6,1,6,-1.,14,9,1,3,2.,9.6977467183023691e-004,0.3375276029109955,0.5532032847404480,0,2,5,10,4,1,-1.,7,10,2,1,2.,5.9512659208849072e-004,0.5631743073463440,0.3359399139881134,0,2,3,8,15,5,-1.,8,8,5,5,3.,-0.1015655994415283,0.0637350380420685,0.5230425000190735,0,2,1,6,5,4,-1.,1,8,5,2,2.,0.0361566990613937,0.5136963129043579,0.1029528975486755,0,2,3,1,17,6,-1.,3,3,17,2,3.,3.4624140243977308e-003,0.3879320025444031,0.5558289289474487,0,2,6,7,8,2,-1.,10,7,4,2,2.,0.0195549800992012,0.5250086784362793,0.1875859946012497,0,2,9,7,3,2,-1.,10,7,1,2,3.,-2.3121440317481756e-003,0.6672028899192810,0.4679641127586365,0,2,8,7,3,2,-1.,9,7,1,2,3.,-1.8605289515107870e-003,0.7163379192352295,0.4334670901298523,0,2,8,9,4,2,-1.,8,10,4,1,2.,-9.4026362057775259e-004,0.3021360933780670,0.5650203227996826,0,2,8,8,4,3,-1.,8,9,4,1,3.,-5.2418331615626812e-003,0.1820009052753449,0.5250256061553955,0,2,9,5,6,4,-1.,9,5,3,4,2.,1.1729019752237946e-004,0.3389188051223755,0.5445973277091980,0,2,8,13,4,3,-1.,8,14,4,1,3.,1.1878840159624815e-003,0.4085349142551422,0.6253563165664673,0,3,4,7,12,6,-1.,10,7,6,3,2.,4,10,6,3,2.,-0.0108813596889377,0.3378399014472961,0.5700082778930664,0,2,8,14,4,3,-1.,8,15,4,1,3.,1.7354859737679362e-003,0.4204635918140411,0.6523038744926453,0,2,9,7,3,3,-1.,9,8,3,1,3.,-6.5119052305817604e-003,0.2595216035842896,0.5428143739700317,0,2,7,4,3,8,-1.,8,4,1,8,3.,-1.2136430013924837e-003,0.6165143847465515,0.3977893888950348,0,2,10,0,3,6,-1.,11,0,1,6,3.,-0.0103542404249310,0.1628028005361557,0.5219504833221436,0,2,6,3,4,8,-1.,8,3,2,8,2.,5.5858830455690622e-004,0.3199650943279266,0.5503574013710022,0,2,14,3,6,13,-1.,14,3,3,13,2.,0.0152996499091387,0.4103994071483612,0.6122388243675232,0,2,8,13,3,6,-1.,8,16,3,3,2.,-0.0215882100164890,0.1034912988543510,0.5197384953498840,0,2,14,3,6,13,-1.,14,3,3,13,2.,-0.1283462941646576,0.8493865132331848,0.4893102943897247,0,3,0,7,10,4,-1.,0,7,5,2,2.,5,9,5,2,2.,-2.2927189711481333e-003,0.3130157887935638,0.5471575260162354,0,2,14,3,6,13,-1.,14,3,3,13,2.,0.0799151062965393,0.4856320917606354,0.6073989272117615,0,2,0,3,6,13,-1.,3,3,3,13,2.,-0.0794410929083824,0.8394674062728882,0.4624533057212830,0,2,9,1,4,1,-1.,9,1,2,1,2.,-5.2800010889768600e-003,0.1881695985794067,0.5306698083877564,0,2,8,0,2,1,-1.,9,0,1,1,2.,1.0463109938427806e-003,0.5271229147911072,0.2583065927028656,0,3,10,16,4,4,-1.,12,16,2,2,2.,10,18,2,2,2.,2.6317298761568964e-004,0.4235304892063141,0.5735440850257874,0,2,9,6,2,3,-1.,10,6,1,3,2.,-3.6173160187900066e-003,0.6934396028518677,0.4495444893836975,0,2,4,5,12,2,-1.,8,5,4,2,3.,0.0114218797534704,0.5900921225547791,0.4138193130493164,0,2,8,7,3,5,-1.,9,7,1,5,3.,-1.9963278900831938e-003,0.6466382741928101,0.4327239990234375,27.1533508300781250,56,0,2,6,4,8,6,-1.,6,6,8,2,3.,-9.9691245704889297e-003,0.6142324209213257,0.2482212036848068,0,2,9,5,2,12,-1.,9,11,2,6,2.,7.3073059320449829e-004,0.5704951882362366,0.2321965992450714,0,2,4,6,6,8,-1.,4,10,6,4,2.,6.4045301405712962e-004,0.2112251967191696,0.5814933180809021,0,2,12,2,8,5,-1.,12,2,4,5,2.,4.5424019917845726e-003,0.2950482070446014,0.5866311788558960,0,2,0,8,18,3,-1.,0,9,18,1,3.,9.2477443104144186e-005,0.2990990877151489,0.5791326761245728,0,2,8,12,4,8,-1.,8,16,4,4,2.,-8.6603146046400070e-003,0.2813029885292053,0.5635542273521423,0,2,0,2,8,5,-1.,4,2,4,5,2.,8.0515816807746887e-003,0.3535369038581848,0.6054757237434387,0,2,13,11,3,4,-1.,13,13,3,2,2.,4.3835240649059415e-004,0.5596532225608826,0.2731510996818543,0,2,5,11,6,1,-1.,7,11,2,1,3.,-9.8168973636347800e-005,0.5978031754493713,0.3638561069965363,0,2,11,3,3,1,-1.,12,3,1,1,3.,-1.1298790341243148e-003,0.2755252122879028,0.5432729125022888,0,2,7,13,5,3,-1.,7,14,5,1,3.,6.4356150105595589e-003,0.4305641949176788,0.7069833278656006,0,2,11,11,7,6,-1.,11,14,7,3,2.,-0.0568293295800686,0.2495242953300476,0.5294997096061707,0,2,2,11,7,6,-1.,2,14,7,3,2.,4.0668169967830181e-003,0.5478553175926209,0.2497723996639252,0,2,12,14,2,6,-1.,12,16,2,2,3.,4.8164798499783501e-005,0.3938601016998291,0.5706356167793274,0,2,8,14,3,3,-1.,8,15,3,1,3.,6.1795017682015896e-003,0.4407606124877930,0.7394766807556152,0,2,11,0,3,5,-1.,12,0,1,5,3.,6.4985752105712891e-003,0.5445243120193481,0.2479152977466583,0,2,6,1,4,9,-1.,8,1,2,9,2.,-1.0211090557277203e-003,0.2544766962528229,0.5338971018791199,0,2,10,3,6,1,-1.,12,3,2,1,3.,-5.4247528314590454e-003,0.2718858122825623,0.5324069261550903,0,2,8,8,3,4,-1.,8,10,3,2,2.,-1.0559899965301156e-003,0.3178288042545319,0.5534508824348450,0,2,8,12,4,2,-1.,8,13,4,1,2.,6.6465808777138591e-004,0.4284219145774841,0.6558194160461426,0,2,5,18,4,2,-1.,5,19,4,1,2.,-2.7524109464138746e-004,0.5902860760688782,0.3810262978076935,0,2,2,1,18,6,-1.,2,3,18,2,3.,4.2293202131986618e-003,0.3816489875316620,0.5709385871887207,0,2,6,0,3,2,-1.,7,0,1,2,3.,-3.2868210691958666e-003,0.1747743934392929,0.5259544253349304,0,3,13,8,6,2,-1.,16,8,3,1,2.,13,9,3,1,2.,1.5611879643984139e-004,0.3601722121238709,0.5725612044334412,0,2,6,10,3,6,-1.,6,13,3,3,2.,-7.3621381488919724e-006,0.5401858091354370,0.3044497072696686,0,3,0,13,20,4,-1.,10,13,10,2,2.,0,15,10,2,2.,-0.0147672500461340,0.3220770061016083,0.5573434829711914,0,2,7,7,6,5,-1.,9,7,2,5,3.,0.0244895908981562,0.4301528036594391,0.6518812775611877,0,2,11,0,2,2,-1.,11,1,2,1,2.,-3.7652091123163700e-004,0.3564583063125610,0.5598236918449402,0,3,1,8,6,2,-1.,1,8,3,1,2.,4,9,3,1,2.,7.3657688517414499e-006,0.3490782976150513,0.5561897754669190,0,3,0,2,20,2,-1.,10,2,10,1,2.,0,3,10,1,2.,-0.0150999398902059,0.1776272058486939,0.5335299968719482,0,2,7,14,5,3,-1.,7,15,5,1,3.,-3.8316650316119194e-003,0.6149687767028809,0.4221394062042236,0,3,7,13,6,6,-1.,10,13,3,3,2.,7,16,3,3,2.,0.0169254001230001,0.5413014888763428,0.2166585028171539,0,2,9,12,2,3,-1.,9,13,2,1,3.,-3.0477850232273340e-003,0.6449490785598755,0.4354617893695831,0,2,16,11,1,6,-1.,16,13,1,2,3.,3.2140589319169521e-003,0.5400155186653137,0.3523217141628265,0,2,3,11,1,6,-1.,3,13,1,2,3.,-4.0023201145231724e-003,0.2774524092674255,0.5338417291641235,0,3,4,4,14,12,-1.,11,4,7,6,2.,4,10,7,6,2.,7.4182129465043545e-003,0.5676739215850830,0.3702817857265472,0,2,5,4,3,3,-1.,5,5,3,1,3.,-8.8764587417244911e-003,0.7749221920967102,0.4583688974380493,0,2,12,3,3,3,-1.,13,3,1,3,3.,2.7311739977449179e-003,0.5338721871376038,0.3996661007404327,0,2,6,6,8,3,-1.,6,7,8,1,3.,-2.5082379579544067e-003,0.5611963272094727,0.3777498900890350,0,2,12,3,3,3,-1.,13,3,1,3,3.,-8.0541074275970459e-003,0.2915228903293610,0.5179182887077332,0,3,3,1,4,10,-1.,3,1,2,5,2.,5,6,2,5,2.,-9.7938813269138336e-004,0.5536432862281799,0.3700192868709564,0,2,5,7,10,2,-1.,5,7,5,2,2.,-5.8745909482240677e-003,0.3754391074180603,0.5679376125335693,0,2,8,7,3,3,-1.,9,7,1,3,3.,-4.4936719350516796e-003,0.7019699215888977,0.4480949938297272,0,2,15,12,2,3,-1.,15,13,2,1,3.,-5.4389229044318199e-003,0.2310364991426468,0.5313386917114258,0,2,7,8,3,4,-1.,8,8,1,4,3.,-7.5094640487805009e-004,0.5864868760108948,0.4129343032836914,0,2,13,4,1,12,-1.,13,10,1,6,2.,1.4528800420521293e-005,0.3732407093048096,0.5619621276855469,0,3,4,5,12,12,-1.,4,5,6,6,2.,10,11,6,6,2.,0.0407580696046352,0.5312091112136841,0.2720521986484528,0,2,7,14,7,3,-1.,7,15,7,1,3.,6.6505931317806244e-003,0.4710015952587128,0.6693493723869324,0,2,3,12,2,3,-1.,3,13,2,1,3.,4.5759351924061775e-003,0.5167819261550903,0.1637275964021683,0,3,3,2,14,2,-1.,10,2,7,1,2.,3,3,7,1,2.,6.5269311890006065e-003,0.5397608876228333,0.2938531935214996,0,2,0,1,3,10,-1.,1,1,1,10,3.,-0.0136603796854615,0.7086488008499146,0.4532200098037720,0,2,9,0,6,5,-1.,11,0,2,5,3.,0.0273588690906763,0.5206481218338013,0.3589231967926025,0,2,5,7,6,2,-1.,8,7,3,2,2.,6.2197551596909761e-004,0.3507075905799866,0.5441123247146606,0,2,7,1,6,10,-1.,7,6,6,5,2.,-3.3077080734074116e-003,0.5859522819519043,0.4024891853332520,0,2,1,1,18,3,-1.,7,1,6,3,3.,-0.0106311095878482,0.6743267178535461,0.4422602951526642,0,2,16,3,3,6,-1.,16,5,3,2,3.,0.0194416493177414,0.5282716155052185,0.1797904968261719,34.5541114807128910,71,0,2,6,3,7,6,-1.,6,6,7,3,2.,-5.5052167735993862e-003,0.5914731025695801,0.2626559138298035,0,2,4,7,12,2,-1.,8,7,4,2,3.,1.9562279339879751e-003,0.2312581986188889,0.5741627216339111,0,2,0,4,17,10,-1.,0,9,17,5,2.,-8.8924784213304520e-003,0.1656530052423477,0.5626654028892517,0,2,3,4,15,16,-1.,3,12,15,8,2.,0.0836383774876595,0.5423449873924255,0.1957294940948486,0,2,7,15,6,4,-1.,7,17,6,2,2.,1.2282270472496748e-003,0.3417904078960419,0.5992503762245178,0,2,15,2,4,9,-1.,15,2,2,9,2.,5.7629169896245003e-003,0.3719581961631775,0.6079903841018677,0,2,2,3,3,2,-1.,2,4,3,1,2.,-1.6417410224676132e-003,0.2577486038208008,0.5576915740966797,0,2,13,6,7,9,-1.,13,9,7,3,3.,3.4113149158656597e-003,0.2950749099254608,0.5514171719551086,0,2,8,11,4,3,-1.,8,12,4,1,3.,-0.0110693201422691,0.7569358944892883,0.4477078914642334,0,3,0,2,20,6,-1.,10,2,10,3,2.,0,5,10,3,2.,0.0348659716546535,0.5583708882331848,0.2669621109962463,0,3,3,2,6,10,-1.,3,2,3,5,2.,6,7,3,5,2.,6.5701099811121821e-004,0.5627313256263733,0.2988890111446381,0,2,13,10,3,4,-1.,13,12,3,2,2.,-0.0243391301482916,0.2771185040473938,0.5108863115310669,0,2,4,10,3,4,-1.,4,12,3,2,2.,5.9435202274471521e-004,0.5580651760101318,0.3120341897010803,0,2,7,5,6,3,-1.,9,5,2,3,3.,2.2971509024500847e-003,0.3330250084400177,0.5679075717926025,0,2,7,6,6,8,-1.,7,10,6,4,2.,-3.7801829166710377e-003,0.2990534901618958,0.5344808101654053,0,2,0,11,20,6,-1.,0,14,20,3,2.,-0.1342066973447800,0.1463858932256699,0.5392568111419678,0,3,4,13,4,6,-1.,4,13,2,3,2.,6,16,2,3,2.,7.5224548345431685e-004,0.3746953904628754,0.5692734718322754,0,3,6,0,8,12,-1.,10,0,4,6,2.,6,6,4,6,2.,-0.0405455417931080,0.2754747867584229,0.5484297871589661,0,2,2,0,15,2,-1.,2,1,15,1,2.,1.2572970008477569e-003,0.3744584023952484,0.5756075978279114,0,2,9,12,2,3,-1.,9,13,2,1,3.,-7.4249948374927044e-003,0.7513859272003174,0.4728231132030487,0,2,3,12,1,2,-1.,3,13,1,1,2.,5.0908129196614027e-004,0.5404896736145020,0.2932321131229401,0,2,9,11,2,3,-1.,9,12,2,1,3.,-1.2808450264856219e-003,0.6169779896736145,0.4273349046707153,0,2,7,3,3,1,-1.,8,3,1,1,3.,-1.8348860321566463e-003,0.2048496007919312,0.5206472277641296,0,2,17,7,3,6,-1.,17,9,3,2,3.,0.0274848695844412,0.5252984762191773,0.1675522029399872,0,2,7,2,3,2,-1.,8,2,1,2,3.,2.2372419480234385e-003,0.5267782807350159,0.2777658104896545,0,2,11,4,5,3,-1.,11,5,5,1,3.,-8.8635291904211044e-003,0.6954557895660400,0.4812048971652985,0,2,4,4,5,3,-1.,4,5,5,1,3.,4.1753971017897129e-003,0.4291887879371643,0.6349195837974548,0,2,19,3,1,2,-1.,19,4,1,1,2.,-1.7098189564421773e-003,0.2930536866188049,0.5361248850822449,0,2,5,5,4,3,-1.,5,6,4,1,3.,6.5328548662364483e-003,0.4495325088500977,0.7409694194793701,0,2,17,7,3,6,-1.,17,9,3,2,3.,-9.5372907817363739e-003,0.3149119913578033,0.5416501760482788,0,2,0,7,3,6,-1.,0,9,3,2,3.,0.0253109894692898,0.5121892094612122,0.1311707943677902,0,2,14,2,6,9,-1.,14,5,6,3,3.,0.0364609695971012,0.5175911784172058,0.2591339945793152,0,2,0,4,5,6,-1.,0,6,5,2,3.,0.0208543296903372,0.5137140154838562,0.1582316011190414,0,2,10,5,6,2,-1.,12,5,2,2,3.,-8.7207747856155038e-004,0.5574309825897217,0.4398978948593140,0,2,4,5,6,2,-1.,6,5,2,2,3.,-1.5227000403683633e-005,0.5548940896987915,0.3708069920539856,0,2,8,1,4,6,-1.,8,3,4,2,3.,-8.4316509310156107e-004,0.3387419879436493,0.5554211139678955,0,2,0,2,3,6,-1.,0,4,3,2,3.,3.6037859972566366e-003,0.5358061790466309,0.3411171138286591,0,2,6,6,8,3,-1.,6,7,8,1,3.,-6.8057891912758350e-003,0.6125202775001526,0.4345862865447998,0,2,0,1,5,9,-1.,0,4,5,3,3.,-0.0470216609537601,0.2358165979385376,0.5193738937377930,0,2,16,0,4,15,-1.,16,0,2,15,2.,-0.0369541086256504,0.7323111295700073,0.4760943949222565,0,2,1,10,3,2,-1.,1,11,3,1,2.,1.0439479956403375e-003,0.5419455170631409,0.3411330878734589,0,2,14,4,1,10,-1.,14,9,1,5,2.,-2.1050689974799752e-004,0.2821694016456604,0.5554947257041931,0,2,0,1,4,12,-1.,2,1,2,12,2.,-0.0808315873146057,0.9129930138587952,0.4697434902191162,0,2,11,11,4,2,-1.,11,11,2,2,2.,-3.6579059087671340e-004,0.6022670269012451,0.3978292942047119,0,2,5,11,4,2,-1.,7,11,2,2,2.,-1.2545920617412776e-004,0.5613213181495667,0.3845539987087250,0,2,3,8,15,5,-1.,8,8,5,5,3.,-0.0687864869832993,0.2261611968278885,0.5300496816635132,0,2,0,0,6,10,-1.,3,0,3,10,2.,0.0124157899990678,0.4075691998004913,0.5828812122344971,0,2,11,4,3,2,-1.,12,4,1,2,3.,-4.7174817882478237e-003,0.2827253937721252,0.5267757773399353,0,2,8,12,3,8,-1.,8,16,3,4,2.,0.0381368584930897,0.5074741244316101,0.1023615971207619,0,2,8,14,5,3,-1.,8,15,5,1,3.,-2.8168049175292253e-003,0.6169006824493408,0.4359692931175232,0,2,7,14,4,3,-1.,7,15,4,1,3.,8.1303603947162628e-003,0.4524433016777039,0.7606095075607300,0,2,11,4,3,2,-1.,12,4,1,2,3.,6.0056019574403763e-003,0.5240408778190613,0.1859712004661560,0,3,3,15,14,4,-1.,3,15,7,2,2.,10,17,7,2,2.,0.0191393196582794,0.5209379196166992,0.2332071959972382,0,3,2,2,16,4,-1.,10,2,8,2,2.,2,4,8,2,2.,0.0164457596838474,0.5450702905654907,0.3264234960079193,0,2,0,8,6,12,-1.,3,8,3,12,2.,-0.0373568907380104,0.6999046802520752,0.4533241987228394,0,2,5,7,10,2,-1.,5,7,5,2,2.,-0.0197279006242752,0.2653664946556091,0.5412809848785400,0,2,9,7,2,5,-1.,10,7,1,5,2.,6.6972579807043076e-003,0.4480566084384918,0.7138652205467224,0,3,13,7,6,4,-1.,16,7,3,2,2.,13,9,3,2,2.,7.4457528535276651e-004,0.4231350123882294,0.5471320152282715,0,2,0,13,8,2,-1.,0,14,8,1,2.,1.1790640419349074e-003,0.5341702103614807,0.3130455017089844,0,3,13,7,6,4,-1.,16,7,3,2,2.,13,9,3,2,2.,0.0349806100130081,0.5118659734725952,0.3430530130863190,0,3,1,7,6,4,-1.,1,7,3,2,2.,4,9,3,2,2.,5.6859792675822973e-004,0.3532187044620514,0.5468639731407166,0,2,12,6,1,12,-1.,12,12,1,6,2.,-0.0113406497985125,0.2842353880405426,0.5348700881004334,0,2,9,5,2,6,-1.,10,5,1,6,2.,-6.6228108480572701e-003,0.6883640289306641,0.4492664933204651,0,2,14,12,2,3,-1.,14,13,2,1,3.,-8.0160330981016159e-003,0.1709893941879273,0.5224308967590332,0,2,4,12,2,3,-1.,4,13,2,1,3.,1.4206819469109178e-003,0.5290846228599548,0.2993383109569550,0,2,8,12,4,3,-1.,8,13,4,1,3.,-2.7801711112260818e-003,0.6498854160308838,0.4460499882698059,0,3,5,2,2,4,-1.,5,2,1,2,2.,6,4,1,2,2.,-1.4747589593753219e-003,0.3260438144207001,0.5388113260269165,0,2,5,5,11,3,-1.,5,6,11,1,3.,-0.0238303393125534,0.7528941035270691,0.4801219999790192,0,2,7,6,4,12,-1.,7,12,4,6,2.,6.9369790144264698e-003,0.5335165858268738,0.3261427879333496,0,2,12,13,8,5,-1.,12,13,4,5,2.,8.2806255668401718e-003,0.4580394029617310,0.5737829804420471,0,2,7,6,1,12,-1.,7,12,1,6,2.,-0.0104395002126694,0.2592320144176483,0.5233827829360962,39.1072883605957030,80,0,2,1,2,6,3,-1.,4,2,3,3,2.,7.2006587870419025e-003,0.3258886039257050,0.6849808096885681,0,3,9,5,6,10,-1.,12,5,3,5,2.,9,10,3,5,2.,-2.8593589086085558e-003,0.5838881134986877,0.2537829875946045,0,3,5,5,8,12,-1.,5,5,4,6,2.,9,11,4,6,2.,6.8580528022721410e-004,0.5708081722259522,0.2812424004077911,0,2,0,7,20,6,-1.,0,9,20,2,3.,7.9580191522836685e-003,0.2501051127910614,0.5544260740280151,0,2,4,2,2,2,-1.,4,3,2,1,2.,-1.2124150525778532e-003,0.2385368049144745,0.5433350205421448,0,2,4,18,12,2,-1.,8,18,4,2,3.,7.9426132142543793e-003,0.3955070972442627,0.6220757961273193,0,2,7,4,4,16,-1.,7,12,4,8,2.,2.4630590341985226e-003,0.5639708042144775,0.2992357909679413,0,2,7,6,7,8,-1.,7,10,7,4,2.,-6.0396599583327770e-003,0.2186512947082520,0.5411676764488220,0,2,6,3,3,1,-1.,7,3,1,1,3.,-1.2988339876756072e-003,0.2350706011056900,0.5364584922790527,0,2,11,15,2,4,-1.,11,17,2,2,2.,2.2299369447864592e-004,0.3804112970829010,0.5729606151580811,0,2,3,5,4,8,-1.,3,9,4,4,2.,1.4654280385002494e-003,0.2510167956352234,0.5258268713951111,0,2,7,1,6,12,-1.,7,7,6,6,2.,-8.1210042117163539e-004,0.5992823839187622,0.3851158916950226,0,2,4,6,6,2,-1.,6,6,2,2,3.,-1.3836020370945334e-003,0.5681396126747131,0.3636586964130402,0,2,16,4,4,6,-1.,16,6,4,2,3.,-0.0279364492744207,0.1491317003965378,0.5377560257911682,0,2,3,3,5,2,-1.,3,4,5,1,2.,-4.6919551095925272e-004,0.3692429959774017,0.5572484731674194,0,2,9,11,2,3,-1.,9,12,2,1,3.,-4.9829659983515739e-003,0.6758509278297424,0.4532504081726074,0,2,2,16,4,2,-1.,2,17,4,1,2.,1.8815309740602970e-003,0.5368022918701172,0.2932539880275726,0,3,7,13,6,6,-1.,10,13,3,3,2.,7,16,3,3,2.,-0.0190675500780344,0.1649377048015595,0.5330067276954651,0,2,7,0,3,4,-1.,8,0,1,4,3.,-4.6906559728085995e-003,0.1963925957679749,0.5119361877441406,0,2,8,15,4,3,-1.,8,16,4,1,3.,5.9777139686048031e-003,0.4671171903610230,0.7008398175239563,0,2,0,4,4,6,-1.,0,6,4,2,3.,-0.0333031304180622,0.1155416965484619,0.5104162096977234,0,2,5,6,12,3,-1.,9,6,4,3,3.,0.0907441079616547,0.5149660110473633,0.1306173056364059,0,2,7,6,6,14,-1.,9,6,2,14,3.,9.3555898638442159e-004,0.3605481088161469,0.5439859032630920,0,2,9,7,3,3,-1.,10,7,1,3,3.,0.0149016501381993,0.4886212050914764,0.7687569856643677,0,2,6,12,2,4,-1.,6,14,2,2,2.,6.1594118596985936e-004,0.5356813073158264,0.3240939080715179,0,2,10,12,7,6,-1.,10,14,7,2,3.,-0.0506709888577461,0.1848621964454651,0.5230404138565064,0,2,1,0,15,2,-1.,1,1,15,1,2.,6.8665749859064817e-004,0.3840579986572266,0.5517945885658264,0,2,14,0,6,6,-1.,14,0,3,6,2.,8.3712432533502579e-003,0.4288564026355743,0.6131753921508789,0,2,5,3,3,1,-1.,6,3,1,1,3.,-1.2953069526702166e-003,0.2913674116134644,0.5280737876892090,0,2,14,0,6,6,-1.,14,0,3,6,2.,-0.0419416800141335,0.7554799914360046,0.4856030941009522,0,2,0,3,20,10,-1.,0,8,20,5,2.,-0.0235293805599213,0.2838279902935028,0.5256081223487854,0,2,14,0,6,6,-1.,14,0,3,6,2.,0.0408574491739273,0.4870935082435608,0.6277297139167786,0,2,0,0,6,6,-1.,3,0,3,6,2.,-0.0254068691283464,0.7099707722663879,0.4575029015541077,0,2,19,15,1,2,-1.,19,16,1,1,2.,-4.1415440500713885e-004,0.4030886888504028,0.5469412207603455,0,2,0,2,4,8,-1.,2,2,2,8,2.,0.0218241196125746,0.4502024054527283,0.6768701076507568,0,3,2,1,18,4,-1.,11,1,9,2,2.,2,3,9,2,2.,0.0141140399500728,0.5442860722541809,0.3791700005531311,0,2,8,12,1,2,-1.,8,13,1,1,2.,6.7214590671937913e-005,0.4200463891029358,0.5873476266860962,0,3,5,2,10,6,-1.,10,2,5,3,2.,5,5,5,3,2.,-7.9417638480663300e-003,0.3792561888694763,0.5585265755653381,0,2,9,7,2,4,-1.,10,7,1,4,2.,-7.2144409641623497e-003,0.7253103852272034,0.4603548943996429,0,2,9,7,3,3,-1.,10,7,1,3,3.,2.5817339774221182e-003,0.4693301916122437,0.5900238752365112,0,2,4,5,12,8,-1.,8,5,4,8,3.,0.1340931951999664,0.5149213075637817,0.1808844953775406,0,2,15,15,4,3,-1.,15,16,4,1,3.,2.2962710354477167e-003,0.5399743914604187,0.3717867136001587,0,2,8,18,3,1,-1.,9,18,1,1,3.,-2.1575849968940020e-003,0.2408495992422104,0.5148863792419434,0,2,9,13,4,3,-1.,9,14,4,1,3.,-4.9196188338100910e-003,0.6573588252067566,0.4738740026950836,0,2,7,13,4,3,-1.,7,14,4,1,3.,1.6267469618469477e-003,0.4192821979522705,0.6303114295005798,0,2,19,15,1,2,-1.,19,16,1,1,2.,3.3413388882763684e-004,0.5540298223495483,0.3702101111412048,0,2,0,15,8,4,-1.,0,17,8,2,2.,-0.0266980808228254,0.1710917949676514,0.5101410746574402,0,2,9,3,6,4,-1.,11,3,2,4,3.,-0.0305618792772293,0.1904218047857285,0.5168793797492981,0,2,8,14,4,3,-1.,8,15,4,1,3.,2.8511548880487680e-003,0.4447506964206696,0.6313853859901428,0,2,3,14,14,6,-1.,3,16,14,2,3.,-0.0362114794552326,0.2490727007389069,0.5377349257469177,0,2,6,3,6,6,-1.,6,6,6,3,2.,-2.4115189444273710e-003,0.5381243228912354,0.3664236962795258,0,2,5,11,10,6,-1.,5,14,10,3,2.,-7.7253201743587852e-004,0.5530232191085815,0.3541550040245056,0,2,3,10,3,4,-1.,4,10,1,4,3.,2.9481729143299162e-004,0.4132699072360992,0.5667243003845215,0,2,13,9,2,2,-1.,13,9,1,2,2.,-6.2334560789167881e-003,0.0987872332334518,0.5198668837547302,0,2,5,3,6,4,-1.,7,3,2,4,3.,-0.0262747295200825,0.0911274924874306,0.5028107166290283,0,2,9,7,3,3,-1.,10,7,1,3,3.,5.3212260827422142e-003,0.4726648926734924,0.6222720742225647,0,2,2,12,2,3,-1.,2,13,2,1,3.,-4.1129058226943016e-003,0.2157457023859024,0.5137804746627808,0,2,9,8,3,12,-1.,9,12,3,4,3.,3.2457809429615736e-003,0.5410770773887634,0.3721776902675629,0,3,3,14,4,6,-1.,3,14,2,3,2.,5,17,2,3,2.,-0.0163597092032433,0.7787874937057495,0.4685291945934296,0,2,16,15,2,2,-1.,16,16,2,1,2.,3.2166109303943813e-004,0.5478987097740173,0.4240373969078064,0,2,2,15,2,2,-1.,2,16,2,1,2.,6.4452440710738301e-004,0.5330560803413391,0.3501324951648712,0,2,8,12,4,3,-1.,8,13,4,1,3.,-7.8909732401371002e-003,0.6923521161079407,0.4726569056510925,0,2,0,7,20,1,-1.,10,7,10,1,2.,0.0483362115919590,0.5055900216102600,0.0757492035627365,0,2,7,6,8,3,-1.,7,6,4,3,2.,-7.5178127735853195e-004,0.3783741891384125,0.5538573861122131,0,2,5,7,8,2,-1.,9,7,4,2,2.,-2.4953910615295172e-003,0.3081651031970978,0.5359612107276917,0,2,9,7,3,5,-1.,10,7,1,5,3.,-2.2385010961443186e-003,0.6633958816528320,0.4649342894554138,0,2,8,7,3,5,-1.,9,7,1,5,3.,-1.7988430336117744e-003,0.6596844792366028,0.4347187876701355,0,2,11,1,3,5,-1.,12,1,1,5,3.,8.7860915809869766e-003,0.5231832861900330,0.2315579950809479,0,2,6,2,3,6,-1.,7,2,1,6,3.,3.6715380847454071e-003,0.5204250216484070,0.2977376878261566,0,2,14,14,6,5,-1.,14,14,3,5,2.,-0.0353364497423172,0.7238878011703491,0.4861505031585693,0,2,9,8,2,2,-1.,9,9,2,1,2.,-6.9189240457490087e-004,0.3105022013187408,0.5229824781417847,0,2,10,7,1,3,-1.,10,8,1,1,3.,-3.3946109469980001e-003,0.3138968050479889,0.5210173726081848,0,3,6,6,2,2,-1.,6,6,1,1,2.,7,7,1,1,2.,9.8569283727556467e-004,0.4536580145359039,0.6585097908973694,0,3,2,11,18,4,-1.,11,11,9,2,2.,2,13,9,2,2.,-0.0501631014049053,0.1804454028606415,0.5198916792869568,0,3,6,6,2,2,-1.,6,6,1,1,2.,7,7,1,1,2.,-2.2367259953171015e-003,0.7255702018737793,0.4651359021663666,0,2,0,15,20,2,-1.,0,16,20,1,2.,7.4326287722215056e-004,0.4412921071052551,0.5898545980453491,0,2,4,14,2,3,-1.,4,15,2,1,3.,-9.3485182151198387e-004,0.3500052988529205,0.5366017818450928,0,2,8,14,4,3,-1.,8,15,4,1,3.,0.0174979399889708,0.4912194907665253,0.8315284848213196,0,2,8,7,2,3,-1.,8,8,2,1,3.,-1.5200000489130616e-003,0.3570275902748108,0.5370560288429260,0,2,9,10,2,3,-1.,9,11,2,1,3.,7.8003940870985389e-004,0.4353772103786469,0.5967335104942322,50.6104812622070310,103,0,2,5,4,10,4,-1.,5,6,10,2,2.,-9.9945552647113800e-003,0.6162583231925964,0.3054533004760742,0,3,9,7,6,4,-1.,12,7,3,2,2.,9,9,3,2,2.,-1.1085229925811291e-003,0.5818294882774353,0.3155578076839447,0,2,4,7,3,6,-1.,4,9,3,2,3.,1.0364380432292819e-003,0.2552052140235901,0.5692911744117737,0,3,11,15,4,4,-1.,13,15,2,2,2.,11,17,2,2,2.,6.8211311008781195e-004,0.3685089945793152,0.5934931039810181,0,2,7,8,4,2,-1.,7,9,4,1,2.,-6.8057340104132891e-004,0.2332392036914825,0.5474792122840881,0,2,13,1,4,3,-1.,13,1,2,3,2.,2.6068789884448051e-004,0.3257457017898560,0.5667545795440674,0,3,5,15,4,4,-1.,5,15,2,2,2.,7,17,2,2,2.,5.1607372006401420e-004,0.3744716942310333,0.5845472812652588,0,2,9,5,4,7,-1.,9,5,2,7,2.,8.5007521556690335e-004,0.3420371115207672,0.5522807240486145,0,2,5,6,8,3,-1.,9,6,4,3,2.,-1.8607829697430134e-003,0.2804419994354248,0.5375424027442932,0,2,9,9,2,2,-1.,9,10,2,1,2.,-1.5033970121294260e-003,0.2579050958156586,0.5498952269554138,0,2,7,15,5,3,-1.,7,16,5,1,3.,2.3478909861296415e-003,0.4175156056880951,0.6313710808753967,0,2,11,10,4,3,-1.,11,10,2,3,2.,-2.8880240279249847e-004,0.5865169763565064,0.4052666127681732,0,2,6,9,8,10,-1.,6,14,8,5,2.,8.9405477046966553e-003,0.5211141109466553,0.2318654060363770,0,2,10,11,6,2,-1.,10,11,3,2,2.,-0.0193277392536402,0.2753432989120483,0.5241525769233704,0,2,4,11,6,2,-1.,7,11,3,2,2.,-2.0202060113660991e-004,0.5722978711128235,0.3677195906639099,0,2,11,3,8,1,-1.,11,3,4,1,2.,2.1179069299250841e-003,0.4466108083724976,0.5542430877685547,0,2,6,3,3,2,-1.,7,3,1,2,3.,-1.7743760254234076e-003,0.2813253104686737,0.5300959944725037,0,2,14,5,6,5,-1.,14,5,3,5,2.,4.2234458960592747e-003,0.4399709999561310,0.5795428156852722,0,2,7,5,2,12,-1.,7,11,2,6,2.,-0.0143752200528979,0.2981117963790894,0.5292059183120728,0,2,8,11,4,3,-1.,8,12,4,1,3.,-0.0153491804376245,0.7705215215682983,0.4748171865940094,0,2,4,1,2,3,-1.,5,1,1,3,2.,1.5152279956964776e-005,0.3718844056129456,0.5576897263526917,0,2,18,3,2,6,-1.,18,5,2,2,3.,-9.1293919831514359e-003,0.3615196049213409,0.5286766886711121,0,2,0,3,2,6,-1.,0,5,2,2,3.,2.2512159775942564e-003,0.5364704728126526,0.3486298024654388,0,2,9,12,2,3,-1.,9,13,2,1,3.,-4.9696918576955795e-003,0.6927651762962341,0.4676836133003235,0,2,7,13,4,3,-1.,7,14,4,1,3.,-0.0128290103748441,0.7712153792381287,0.4660735130310059,0,2,18,0,2,6,-1.,18,2,2,2,3.,-9.3660065904259682e-003,0.3374983966350555,0.5351287722587585,0,2,0,0,2,6,-1.,0,2,2,2,3.,3.2452319283038378e-003,0.5325189828872681,0.3289610147476196,0,2,8,14,6,3,-1.,8,15,6,1,3.,-0.0117235602810979,0.6837652921676636,0.4754300117492676,0,2,7,4,2,4,-1.,8,4,1,4,2.,2.9257940695970319e-005,0.3572087883949280,0.5360502004623413,0,2,8,5,4,6,-1.,8,7,4,2,3.,-2.2244219508138485e-005,0.5541427135467529,0.3552064001560211,0,2,6,4,2,2,-1.,7,4,1,2,2.,5.0881509669125080e-003,0.5070844292640686,0.1256462037563324,0,3,3,14,14,4,-1.,10,14,7,2,2.,3,16,7,2,2.,0.0274296794086695,0.5269560217857361,0.1625818014144898,0,3,6,15,6,2,-1.,6,15,3,1,2.,9,16,3,1,2.,-6.4142867922782898e-003,0.7145588994026184,0.4584197103977203,0,2,14,15,6,2,-1.,14,16,6,1,2.,3.3479959238320589e-003,0.5398612022399902,0.3494696915149689,0,2,2,12,12,8,-1.,2,16,12,4,2.,-0.0826354920864105,0.2439192980527878,0.5160226225852966,0,2,7,7,7,2,-1.,7,8,7,1,2.,1.0261740535497665e-003,0.3886891901493073,0.5767908096313477,0,2,0,2,18,2,-1.,0,3,18,1,2.,-1.6307090409100056e-003,0.3389458060264587,0.5347700715065002,0,2,9,6,2,5,-1.,9,6,1,5,2.,2.4546680506318808e-003,0.4601413905620575,0.6387246847152710,0,2,7,5,3,8,-1.,8,5,1,8,3.,-9.9476519972085953e-004,0.5769879221916199,0.4120396077632904,0,2,9,6,3,4,-1.,10,6,1,4,3.,0.0154091902077198,0.4878709018230438,0.7089822292327881,0,2,4,13,3,2,-1.,4,14,3,1,2.,1.1784400558099151e-003,0.5263553261756897,0.2895244956016541,0,2,9,4,6,3,-1.,11,4,2,3,3.,-0.0277019198983908,0.1498828977346420,0.5219606757164002,0,2,5,4,6,3,-1.,7,4,2,3,3.,-0.0295053999871016,0.0248933192342520,0.4999816119670868,0,2,14,11,5,2,-1.,14,12,5,1,2.,4.5159430010244250e-004,0.5464622974395752,0.4029662907123566,0,2,1,2,6,9,-1.,3,2,2,9,3.,7.1772639639675617e-003,0.4271056950092316,0.5866296887397766,0,2,14,6,6,13,-1.,14,6,3,13,2.,-0.0741820484399796,0.6874179244041443,0.4919027984142304,0,3,3,6,14,8,-1.,3,6,7,4,2.,10,10,7,4,2.,-0.0172541607171297,0.3370676040649414,0.5348739027976990,0,2,16,0,4,11,-1.,16,0,2,11,2.,0.0148515598848462,0.4626792967319489,0.6129904985427856,0,3,3,4,12,12,-1.,3,4,6,6,2.,9,10,6,6,2.,0.0100020002573729,0.5346122980117798,0.3423453867435455,0,2,11,4,5,3,-1.,11,5,5,1,3.,2.0138120744377375e-003,0.4643830060958862,0.5824304223060608,0,2,4,11,4,2,-1.,4,12,4,1,2.,1.5135470312088728e-003,0.5196396112442017,0.2856149971485138,0,2,10,7,2,2,-1.,10,7,1,2,2.,3.1381431035697460e-003,0.4838162958621979,0.5958529710769653,0,2,8,7,2,2,-1.,9,7,1,2,2.,-5.1450440660119057e-003,0.8920302987098694,0.4741412103176117,0,2,9,17,3,2,-1.,10,17,1,2,3.,-4.4736708514392376e-003,0.2033942937850952,0.5337278842926025,0,2,5,6,3,3,-1.,5,7,3,1,3.,1.9628470763564110e-003,0.4571633934974670,0.6725863218307495,0,2,10,0,3,3,-1.,11,0,1,3,3.,5.4260450415313244e-003,0.5271108150482178,0.2845670878887177,0,3,5,6,6,2,-1.,5,6,3,1,2.,8,7,3,1,2.,4.9611460417509079e-004,0.4138312935829163,0.5718597769737244,0,2,12,16,4,3,-1.,12,17,4,1,3.,9.3728788197040558e-003,0.5225151181221008,0.2804847061634064,0,2,3,12,3,2,-1.,3,13,3,1,2.,6.0500897234305739e-004,0.5236768722534180,0.3314523994922638,0,2,9,12,3,2,-1.,9,13,3,1,2.,5.6792551185935736e-004,0.4531059861183167,0.6276971101760864,0,3,1,11,16,4,-1.,1,11,8,2,2.,9,13,8,2,2.,0.0246443394571543,0.5130851864814758,0.2017143964767456,0,2,12,4,3,3,-1.,12,5,3,1,3.,-0.0102904504165053,0.7786595225334168,0.4876641035079956,0,2,4,4,5,3,-1.,4,5,5,1,3.,2.0629419013857841e-003,0.4288598895072937,0.5881264209747315,0,2,12,16,4,3,-1.,12,17,4,1,3.,-5.0519481301307678e-003,0.3523977994918823,0.5286008715629578,0,2,5,4,3,3,-1.,5,5,3,1,3.,-5.7692620903253555e-003,0.6841086149215698,0.4588094055652618,0,2,9,0,2,2,-1.,9,1,2,1,2.,-4.5789941214025021e-004,0.3565520048141480,0.5485978126525879,0,2,8,9,4,2,-1.,8,10,4,1,2.,-7.5918837683275342e-004,0.3368793129920960,0.5254197120666504,0,2,8,8,4,3,-1.,8,9,4,1,3.,-1.7737259622663260e-003,0.3422161042690277,0.5454015135765076,0,2,0,13,6,3,-1.,2,13,2,3,3.,-8.5610467940568924e-003,0.6533612012863159,0.4485856890678406,0,2,16,14,3,2,-1.,16,15,3,1,2.,1.7277270089834929e-003,0.5307580232620239,0.3925352990627289,0,2,1,18,18,2,-1.,7,18,6,2,3.,-0.0281996093690395,0.6857458949089050,0.4588584005832672,0,2,16,14,3,2,-1.,16,15,3,1,2.,-1.7781109781935811e-003,0.4037851095199585,0.5369856953620911,0,2,1,14,3,2,-1.,1,15,3,1,2.,3.3177141449414194e-004,0.5399798750877380,0.3705750107765198,0,2,7,14,6,3,-1.,7,15,6,1,3.,2.6385399978607893e-003,0.4665437042713165,0.6452730894088745,0,2,5,14,8,3,-1.,5,15,8,1,3.,-2.1183069329708815e-003,0.5914781093597412,0.4064677059650421,0,2,10,6,4,14,-1.,10,6,2,14,2.,-0.0147732896730304,0.3642038106918335,0.5294762849807739,0,2,6,6,4,14,-1.,8,6,2,14,2.,-0.0168154407292604,0.2664231956005096,0.5144972801208496,0,2,13,5,2,3,-1.,13,6,2,1,3.,-6.3370140269398689e-003,0.6779531240463257,0.4852097928524017,0,2,7,16,6,1,-1.,9,16,2,1,3.,-4.4560048991115764e-005,0.5613964796066284,0.4153054058551788,0,2,9,12,3,3,-1.,9,13,3,1,3.,-1.0240620467811823e-003,0.5964478254318237,0.4566304087638855,0,2,7,0,3,3,-1.,8,0,1,3,3.,-2.3161689750850201e-003,0.2976115047931671,0.5188159942626953,0,2,4,0,16,18,-1.,4,9,16,9,2.,0.5321757197380066,0.5187839269638062,0.2202631980180740,0,2,1,1,16,14,-1.,1,8,16,7,2.,-0.1664305031299591,0.1866022944450378,0.5060343146324158,0,2,3,9,15,4,-1.,8,9,5,4,3.,0.1125352978706360,0.5212125182151794,0.1185022965073586,0,2,6,12,7,3,-1.,6,13,7,1,3.,9.3046864494681358e-003,0.4589937031269074,0.6826149225234985,0,2,14,15,2,3,-1.,14,16,2,1,3.,-4.6255099587142467e-003,0.3079940974712372,0.5225008726119995,0,3,2,3,16,14,-1.,2,3,8,7,2.,10,10,8,7,2.,-0.1111646965146065,0.2101044058799744,0.5080801844596863,0,3,16,2,4,18,-1.,18,2,2,9,2.,16,11,2,9,2.,-0.0108884396031499,0.5765355229377747,0.4790464043617249,0,2,4,15,2,3,-1.,4,16,2,1,3.,5.8564301580190659e-003,0.5065100193023682,0.1563598960638046,0,3,16,2,4,18,-1.,18,2,2,9,2.,16,11,2,9,2.,0.0548543892800808,0.4966914951801300,0.7230510711669922,0,2,1,1,8,3,-1.,1,2,8,1,3.,-0.0111973397433758,0.2194979041814804,0.5098798274993897,0,2,8,11,4,3,-1.,8,12,4,1,3.,4.4069071300327778e-003,0.4778401851654053,0.6770902872085571,0,2,5,11,5,9,-1.,5,14,5,3,3.,-0.0636652931571007,0.1936362981796265,0.5081024169921875,0,2,16,0,4,11,-1.,16,0,2,11,2.,-9.8081491887569427e-003,0.5999063253402710,0.4810341000556946,0,2,7,0,6,1,-1.,9,0,2,1,3.,-2.1717099007219076e-003,0.3338333964347839,0.5235472917556763,0,2,16,3,3,7,-1.,17,3,1,7,3.,-0.0133155202493072,0.6617069840431213,0.4919213056564331,0,2,1,3,3,7,-1.,2,3,1,7,3.,2.5442079640924931e-003,0.4488744139671326,0.6082184910774231,0,2,7,8,6,12,-1.,7,12,6,4,3.,0.0120378397405148,0.5409392118453980,0.3292432129383087,0,2,0,0,4,11,-1.,2,0,2,11,2.,-0.0207010507583618,0.6819120049476624,0.4594995975494385,0,2,14,0,6,20,-1.,14,0,3,20,2.,0.0276082791388035,0.4630792140960693,0.5767282843589783,0,2,0,3,1,2,-1.,0,4,1,1,2.,1.2370620388537645e-003,0.5165379047393799,0.2635016143321991,0,3,5,5,10,8,-1.,10,5,5,4,2.,5,9,5,4,2.,-0.0376693382859230,0.2536393105983734,0.5278980135917664,0,3,4,7,12,4,-1.,4,7,6,2,2.,10,9,6,2,2.,-1.8057259730994701e-003,0.3985156118869782,0.5517500042915344,54.6200714111328130,111,0,2,2,1,6,4,-1.,5,1,3,4,2.,4.4299028813838959e-003,0.2891018092632294,0.6335226297378540,0,3,9,7,6,4,-1.,12,7,3,2,2.,9,9,3,2,2.,-2.3813319858163595e-003,0.6211789250373840,0.3477487862110138,0,2,5,6,2,6,-1.,5,9,2,3,2.,2.2915711160749197e-003,0.2254412025213242,0.5582118034362793,0,3,9,16,6,4,-1.,12,16,3,2,2.,9,18,3,2,2.,9.9457940086722374e-004,0.3711710870265961,0.5930070877075195,0,2,9,4,2,12,-1.,9,10,2,6,2.,7.7164667891338468e-004,0.5651720166206360,0.3347995877265930,0,2,7,1,6,18,-1.,9,1,2,18,3.,-1.1386410333216190e-003,0.3069126009941101,0.5508630871772766,0,2,4,12,12,2,-1.,8,12,4,2,3.,-1.6403039626311511e-004,0.5762827992439270,0.3699047863483429,0,2,8,8,6,2,-1.,8,9,6,1,2.,2.9793529392918572e-005,0.2644244134426117,0.5437911152839661,0,2,8,0,3,6,-1.,9,0,1,6,3.,8.5774902254343033e-003,0.5051138997077942,0.1795724928379059,0,2,11,18,3,2,-1.,11,19,3,1,2.,-2.6032689493149519e-004,0.5826969146728516,0.4446826875209808,0,2,1,1,17,4,-1.,1,3,17,2,2.,-6.1404630541801453e-003,0.3113852143287659,0.5346971750259399,0,2,11,8,4,12,-1.,11,8,2,12,2.,-0.0230869501829147,0.3277946114540100,0.5331197977066040,0,2,8,14,4,3,-1.,8,15,4,1,3.,-0.0142436502501369,0.7381709814071655,0.4588063061237335,0,2,12,3,2,17,-1.,12,3,1,17,2.,0.0194871295243502,0.5256630778312683,0.2274471968412399,0,2,4,7,6,1,-1.,6,7,2,1,3.,-9.6681108698248863e-004,0.5511230826377869,0.3815006911754608,0,2,18,3,2,3,-1.,18,4,2,1,3.,3.1474709976464510e-003,0.5425636768341065,0.2543726861476898,0,2,8,4,3,4,-1.,8,6,3,2,2.,-1.8026070029009134e-004,0.5380191802978516,0.3406304121017456,0,2,4,5,12,10,-1.,4,10,12,5,2.,-6.0266260989010334e-003,0.3035801947116852,0.5420572161674500,0,2,5,18,4,2,-1.,7,18,2,2,2.,4.4462960795499384e-004,0.3990997076034546,0.5660110116004944,0,2,17,2,3,6,-1.,17,4,3,2,3.,2.2609760053455830e-003,0.5562806725502014,0.3940688073635101,0,2,7,7,6,6,-1.,9,7,2,6,3.,0.0511330589652061,0.4609653949737549,0.7118561863899231,0,2,17,2,3,6,-1.,17,4,3,2,3.,-0.0177863091230392,0.2316166013479233,0.5322144031524658,0,2,8,0,3,4,-1.,9,0,1,4,3.,-4.9679628573358059e-003,0.2330771982669830,0.5122029185295105,0,2,9,14,2,3,-1.,9,15,2,1,3.,2.0667689386755228e-003,0.4657444059848785,0.6455488204956055,0,2,0,12,6,3,-1.,0,13,6,1,3.,7.4413768015801907e-003,0.5154392123222351,0.2361633926630020,0,2,8,14,4,3,-1.,8,15,4,1,3.,-3.6277279723435640e-003,0.6219773292541504,0.4476661086082459,0,2,3,12,2,3,-1.,3,13,2,1,3.,-5.3530759178102016e-003,0.1837355047464371,0.5102208256721497,0,2,5,6,12,7,-1.,9,6,4,7,3.,0.1453091949224472,0.5145987272262573,0.1535930931568146,0,2,0,2,3,6,-1.,0,4,3,2,3.,2.4394490756094456e-003,0.5343660116195679,0.3624661862850189,0,2,14,6,1,3,-1.,14,7,1,1,3.,-3.1283390708267689e-003,0.6215007901191711,0.4845592081546783,0,2,2,0,3,14,-1.,3,0,1,14,3.,1.7940260004252195e-003,0.4299261868000031,0.5824198126792908,0,2,12,14,5,6,-1.,12,16,5,2,3.,0.0362538211047649,0.5260334014892578,0.1439467966556549,0,2,4,14,5,6,-1.,4,16,5,2,3.,-5.1746722310781479e-003,0.3506538867950440,0.5287045240402222,0,3,11,10,2,2,-1.,12,10,1,1,2.,11,11,1,1,2.,6.5383297624066472e-004,0.4809640944004059,0.6122040152549744,0,2,5,0,3,14,-1.,6,0,1,14,3.,-0.0264802295714617,0.1139362007379532,0.5045586228370667,0,2,10,15,2,3,-1.,10,16,2,1,3.,-3.0440660193562508e-003,0.6352095007896423,0.4794734120368958,0,2,0,2,2,3,-1.,0,3,2,1,3.,3.6993520334362984e-003,0.5131118297576904,0.2498510926961899,0,2,5,11,12,6,-1.,5,14,12,3,2.,-3.6762931267730892e-004,0.5421394705772400,0.3709532022476196,0,2,6,11,3,9,-1.,6,14,3,3,3.,-0.0413822606205940,0.1894959956407547,0.5081691741943359,0,3,11,10,2,2,-1.,12,10,1,1,2.,11,11,1,1,2.,-1.0532729793339968e-003,0.6454367041587830,0.4783608913421631,0,2,5,6,1,3,-1.,5,7,1,1,3.,-2.1648600231856108e-003,0.6215031147003174,0.4499826133251190,0,2,4,9,13,3,-1.,4,10,13,1,3.,-5.6747748749330640e-004,0.3712610900402069,0.5419334769248962,0,2,1,7,15,6,-1.,6,7,5,6,3.,0.1737584024667740,0.5023643970489502,0.1215742006897926,0,2,4,5,12,6,-1.,8,5,4,6,3.,-2.9049699660390615e-003,0.3240267932415009,0.5381883978843689,0,2,8,10,4,3,-1.,8,11,4,1,3.,1.2299539521336555e-003,0.4165507853031158,0.5703486204147339,0,2,15,14,1,3,-1.,15,15,1,1,3.,-5.4329237900674343e-004,0.3854042887687683,0.5547549128532410,0,2,1,11,5,3,-1.,1,12,5,1,3.,-8.3297258242964745e-003,0.2204494029283524,0.5097082853317261,0,2,7,1,7,12,-1.,7,7,7,6,2.,-1.0417630255687982e-004,0.5607066154479981,0.4303036034107208,0,3,0,1,6,10,-1.,0,1,3,5,2.,3,6,3,5,2.,0.0312047004699707,0.4621657133102417,0.6982004046440125,0,2,16,1,4,3,-1.,16,2,4,1,3.,7.8943502157926559e-003,0.5269594192504883,0.2269068062305450,0,2,5,5,2,3,-1.,5,6,2,1,3.,-4.3645310215651989e-003,0.6359223127365112,0.4537956118583679,0,2,12,2,3,5,-1.,13,2,1,5,3.,7.6793059706687927e-003,0.5274767875671387,0.2740483880043030,0,2,0,3,4,6,-1.,0,5,4,2,3.,-0.0254311393946409,0.2038519978523254,0.5071732997894287,0,2,8,12,4,2,-1.,8,13,4,1,2.,8.2000601105391979e-004,0.4587455093860626,0.6119868159294128,0,2,8,18,3,1,-1.,9,18,1,1,3.,2.9284600168466568e-003,0.5071274042129517,0.2028204947710037,0,3,11,10,2,2,-1.,12,10,1,1,2.,11,11,1,1,2.,4.5256470912136137e-005,0.4812104105949402,0.5430821776390076,0,3,7,10,2,2,-1.,7,10,1,1,2.,8,11,1,1,2.,1.3158309739083052e-003,0.4625813961029053,0.6779323220252991,0,2,11,11,4,4,-1.,11,13,4,2,2.,1.5870389761403203e-003,0.5386291742324829,0.3431465029716492,0,2,8,12,3,8,-1.,9,12,1,8,3.,-0.0215396601706743,0.0259425006806850,0.5003222823143005,0,2,13,0,6,3,-1.,13,1,6,1,3.,0.0143344802781940,0.5202844738960266,0.1590632945299149,0,2,8,8,3,4,-1.,9,8,1,4,3.,-8.3881383761763573e-003,0.7282481193542481,0.4648044109344482,0,3,5,7,10,10,-1.,10,7,5,5,2.,5,12,5,5,2.,9.1906841844320297e-003,0.5562356710433960,0.3923191130161285,0,3,3,18,8,2,-1.,3,18,4,1,2.,7,19,4,1,2.,-5.8453059755265713e-003,0.6803392767906189,0.4629127979278565,0,2,10,2,6,8,-1.,12,2,2,8,3.,-0.0547077991068363,0.2561671137809753,0.5206125974655151,0,2,4,2,6,8,-1.,6,2,2,8,3.,9.1142775490880013e-003,0.5189620256423950,0.3053877055644989,0,2,11,0,3,7,-1.,12,0,1,7,3.,-0.0155750000849366,0.1295074969530106,0.5169094800949097,0,2,7,11,2,1,-1.,8,11,1,1,2.,-1.2050600344082341e-004,0.5735098123550415,0.4230825006961823,0,2,15,14,1,3,-1.,15,15,1,1,3.,1.2273970060050488e-003,0.5289878249168396,0.4079791903495789,0,3,7,15,2,2,-1.,7,15,1,1,2.,8,16,1,1,2.,-1.2186600361019373e-003,0.6575639843940735,0.4574409127235413,0,2,15,14,1,3,-1.,15,15,1,1,3.,-3.3256649039685726e-003,0.3628047108650208,0.5195019841194153,0,2,6,0,3,7,-1.,7,0,1,7,3.,-0.0132883097976446,0.1284265965223312,0.5043488740921021,0,2,18,1,2,7,-1.,18,1,1,7,2.,-3.3839771058410406e-003,0.6292240023612976,0.4757505953311920,0,2,2,0,8,20,-1.,2,10,8,10,2.,-0.2195422053337097,0.1487731933593750,0.5065013766288757,0,2,3,0,15,6,-1.,3,2,15,2,3.,4.9111708067357540e-003,0.4256102144718170,0.5665838718414307,0,2,4,3,12,2,-1.,4,4,12,1,2.,-1.8744950648397207e-004,0.4004144072532654,0.5586857199668884,0,2,16,0,4,5,-1.,16,0,2,5,2.,-5.2178641781210899e-003,0.6009116172790527,0.4812706112861633,0,2,7,0,3,4,-1.,8,0,1,4,3.,-1.1111519997939467e-003,0.3514933884143829,0.5287089943885803,0,2,16,0,4,5,-1.,16,0,2,5,2.,4.4036400504410267e-003,0.4642275869846344,0.5924085974693298,0,2,1,7,6,13,-1.,3,7,2,13,3.,0.1229949966073036,0.5025529265403748,0.0691524818539619,0,2,16,0,4,5,-1.,16,0,2,5,2.,-0.0123135102912784,0.5884591937065125,0.4934012889862061,0,2,0,0,4,5,-1.,2,0,2,5,2.,4.1471039876341820e-003,0.4372239112854004,0.5893477797508240,0,2,14,12,3,6,-1.,14,14,3,2,3.,-3.5502649843692780e-003,0.4327551126480103,0.5396270155906677,0,2,3,12,3,6,-1.,3,14,3,2,3.,-0.0192242693156004,0.1913134008646011,0.5068330764770508,0,2,16,1,4,3,-1.,16,2,4,1,3.,1.4395059552043676e-003,0.5308178067207336,0.4243533015251160,0,3,8,7,2,10,-1.,8,7,1,5,2.,9,12,1,5,2.,-6.7751999013125896e-003,0.6365395784378052,0.4540086090564728,0,2,11,11,4,4,-1.,11,13,4,2,2.,7.0119630545377731e-003,0.5189834237098694,0.3026199936866760,0,2,0,1,4,3,-1.,0,2,4,1,3.,5.4014651104807854e-003,0.5105062127113342,0.2557682991027832,0,2,13,4,1,3,-1.,13,5,1,1,3.,9.0274988906458020e-004,0.4696914851665497,0.5861827731132507,0,2,7,15,3,5,-1.,8,15,1,5,3.,0.0114744501188397,0.5053645968437195,0.1527177989482880,0,2,9,7,3,5,-1.,10,7,1,5,3.,-6.7023430019617081e-003,0.6508980989456177,0.4890604019165039,0,2,8,7,3,5,-1.,9,7,1,5,3.,-2.0462959073483944e-003,0.6241816878318787,0.4514600038528442,0,2,10,6,4,14,-1.,10,6,2,14,2.,-9.9951568990945816e-003,0.3432781100273132,0.5400953888893127,0,2,0,5,5,6,-1.,0,7,5,2,3.,-0.0357007086277008,0.1878059059381485,0.5074077844619751,0,2,9,5,6,4,-1.,9,5,3,4,2.,4.5584561303257942e-004,0.3805277049541473,0.5402569770812988,0,2,0,0,18,10,-1.,6,0,6,10,3.,-0.0542606003582478,0.6843714714050293,0.4595097005367279,0,2,10,6,4,14,-1.,10,6,2,14,2.,6.0600461438298225e-003,0.5502905249595642,0.4500527977943420,0,2,6,6,4,14,-1.,8,6,2,14,2.,-6.4791832119226456e-003,0.3368858098983765,0.5310757160186768,0,2,13,4,1,3,-1.,13,5,1,1,3.,-1.4939469983801246e-003,0.6487640142440796,0.4756175875663757,0,2,5,1,2,3,-1.,6,1,1,3,2.,1.4610530342906713e-005,0.4034579098224640,0.5451064109802246,0,3,18,1,2,18,-1.,19,1,1,9,2.,18,10,1,9,2.,-7.2321938350796700e-003,0.6386873722076416,0.4824739992618561,0,2,2,1,4,3,-1.,2,2,4,1,3.,-4.0645818226039410e-003,0.2986421883106232,0.5157335996627808,0,3,18,1,2,18,-1.,19,1,1,9,2.,18,10,1,9,2.,0.0304630808532238,0.5022199749946594,0.7159956097602844,0,3,1,14,4,6,-1.,1,14,2,3,2.,3,17,2,3,2.,-8.0544911324977875e-003,0.6492452025413513,0.4619275033473969,0,2,10,11,7,6,-1.,10,13,7,2,3.,0.0395051389932632,0.5150570869445801,0.2450613975524902,0,3,0,10,6,10,-1.,0,10,3,5,2.,3,15,3,5,2.,8.4530208259820938e-003,0.4573669135570526,0.6394037008285523,0,2,11,0,3,4,-1.,12,0,1,4,3.,-1.1688120430335402e-003,0.3865512013435364,0.5483661293983460,0,2,5,10,5,6,-1.,5,13,5,3,2.,2.8070670086890459e-003,0.5128579139709473,0.2701480090618134,0,2,14,6,1,8,-1.,14,10,1,4,2.,4.7365209320560098e-004,0.4051581919193268,0.5387461185455322,0,3,1,7,18,6,-1.,1,7,9,3,2.,10,10,9,3,2.,0.0117410803213716,0.5295950174331665,0.3719413876533508,0,2,9,7,2,2,-1.,9,7,1,2,2.,3.1833238899707794e-003,0.4789406955242157,0.6895126104354858,0,2,5,9,4,5,-1.,7,9,2,5,2.,7.0241501089185476e-004,0.5384489297866821,0.3918080925941467,50.1697311401367190,102,0,2,7,6,6,3,-1.,9,6,2,3,3.,0.0170599296689034,0.3948527872562408,0.7142534852027893,0,2,1,0,18,4,-1.,7,0,6,4,3.,0.0218408405780792,0.3370316028594971,0.6090016961097717,0,2,7,15,2,4,-1.,7,17,2,2,2.,2.4520049919374287e-004,0.3500576019287109,0.5987902283668518,0,2,1,0,19,9,-1.,1,3,19,3,3.,8.3272606134414673e-003,0.3267528116703033,0.5697240829467773,0,2,3,7,3,6,-1.,3,9,3,2,3.,5.7148298947140574e-004,0.3044599890708923,0.5531656742095947,0,3,13,7,4,4,-1.,15,7,2,2,2.,13,9,2,2,2.,6.7373987985774875e-004,0.3650012016296387,0.5672631263732910,0,3,3,7,4,4,-1.,3,7,2,2,2.,5,9,2,2,2.,3.4681590477703139e-005,0.3313541114330292,0.5388727188110352,0,2,9,6,10,8,-1.,9,10,10,4,2.,-5.8563398197293282e-003,0.2697942852973938,0.5498778820037842,0,2,3,8,14,12,-1.,3,14,14,6,2.,8.5102273151278496e-003,0.5269358158111572,0.2762879133224487,0,3,6,5,10,12,-1.,11,5,5,6,2.,6,11,5,6,2.,-0.0698172077536583,0.2909603118896484,0.5259246826171875,0,2,9,11,2,3,-1.,9,12,2,1,3.,-8.6113670840859413e-004,0.5892577171325684,0.4073697924613953,0,2,9,5,6,5,-1.,9,5,3,5,2.,9.7149249631911516e-004,0.3523564040660858,0.5415862202644348,0,2,9,4,2,4,-1.,9,6,2,2,2.,-1.4727490452060010e-005,0.5423017740249634,0.3503156006336212,0,2,9,5,6,5,-1.,9,5,3,5,2.,0.0484202913939953,0.5193945765495300,0.3411195874214172,0,2,5,5,6,5,-1.,8,5,3,5,2.,1.3257140526548028e-003,0.3157769143581390,0.5335376262664795,0,2,11,2,6,1,-1.,13,2,2,1,3.,1.4922149603080470e-005,0.4451299905776978,0.5536553859710693,0,2,3,2,6,1,-1.,5,2,2,1,3.,-2.7173398993909359e-003,0.3031741976737976,0.5248088836669922,0,2,13,5,2,3,-1.,13,6,2,1,3.,2.9219500720500946e-003,0.4781453013420105,0.6606041789054871,0,2,0,10,1,4,-1.,0,12,1,2,2.,-1.9804988987743855e-003,0.3186308145523071,0.5287625193595886,0,2,13,5,2,3,-1.,13,6,2,1,3.,-4.0012109093368053e-003,0.6413596868515015,0.4749928116798401,0,2,8,18,3,2,-1.,9,18,1,2,3.,-4.3491991236805916e-003,0.1507498025894165,0.5098996758460999,0,2,6,15,9,2,-1.,6,16,9,1,2.,1.3490889687091112e-003,0.4316158890724182,0.5881167054176331,0,2,8,14,4,3,-1.,8,15,4,1,3.,0.0185970701277256,0.4735553860664368,0.9089794158935547,0,2,18,4,2,4,-1.,18,6,2,2,2.,-1.8562379991635680e-003,0.3553189039230347,0.5577837228775024,0,2,5,5,2,3,-1.,5,6,2,1,3.,2.2940430790185928e-003,0.4500094950199127,0.6580877900123596,0,2,15,16,3,2,-1.,15,17,3,1,2.,2.9982850537635386e-004,0.5629242062568665,0.3975878953933716,0,2,0,0,3,9,-1.,0,3,3,3,3.,3.5455459728837013e-003,0.5381547212600708,0.3605485856533051,0,2,9,7,3,3,-1.,9,8,3,1,3.,9.6104722470045090e-003,0.5255997180938721,0.1796745955944061,0,2,8,7,3,3,-1.,8,8,3,1,3.,-6.2783220782876015e-003,0.2272856980562210,0.5114030241966248,0,2,9,5,2,6,-1.,9,5,1,6,2.,3.4598479978740215e-003,0.4626308083534241,0.6608219146728516,0,2,8,6,3,4,-1.,9,6,1,4,3.,-1.3112019514665008e-003,0.6317539811134338,0.4436857998371124,0,3,7,6,8,12,-1.,11,6,4,6,2.,7,12,4,6,2.,2.6876179035753012e-003,0.5421109795570374,0.4054022133350372,0,3,5,6,8,12,-1.,5,6,4,6,2.,9,12,4,6,2.,3.9118169806897640e-003,0.5358477830886841,0.3273454904556274,0,2,12,4,3,3,-1.,12,5,3,1,3.,-0.0142064504325390,0.7793576717376709,0.4975781142711639,0,2,2,16,3,2,-1.,2,17,3,1,2.,7.1705528534948826e-004,0.5297319889068604,0.3560903966426849,0,2,12,4,3,3,-1.,12,5,3,1,3.,1.6635019565001130e-003,0.4678094089031220,0.5816481709480286,0,2,2,12,6,6,-1.,2,14,6,2,3.,3.3686188980937004e-003,0.5276734232902527,0.3446420133113861,0,2,7,13,6,3,-1.,7,14,6,1,3.,0.0127995302900672,0.4834679961204529,0.7472159266471863,0,2,6,14,6,3,-1.,6,15,6,1,3.,3.3901201095432043e-003,0.4511859118938446,0.6401721239089966,0,2,14,15,5,3,-1.,14,16,5,1,3.,4.7070779837667942e-003,0.5335658788681030,0.3555220961570740,0,2,5,4,3,3,-1.,5,5,3,1,3.,1.4819339849054813e-003,0.4250707030296326,0.5772724151611328,0,2,14,15,5,3,-1.,14,16,5,1,3.,-6.9995759986341000e-003,0.3003320097923279,0.5292900204658508,0,2,5,3,6,2,-1.,7,3,2,2,3.,0.0159390103071928,0.5067319273948669,0.1675581932067871,0,2,8,15,4,3,-1.,8,16,4,1,3.,7.6377349905669689e-003,0.4795069992542267,0.7085601091384888,0,2,1,15,5,3,-1.,1,16,5,1,3.,6.7334040068089962e-003,0.5133113265037537,0.2162470072507858,0,3,8,13,4,6,-1.,10,13,2,3,2.,8,16,2,3,2.,-0.0128588099032640,0.1938841938972473,0.5251371860504150,0,2,7,8,3,3,-1.,8,8,1,3,3.,-6.2270800117403269e-004,0.5686538219451904,0.4197868108749390,0,2,12,0,5,4,-1.,12,2,5,2,2.,-5.2651681471616030e-004,0.4224168956279755,0.5429695844650269,0,3,0,2,20,2,-1.,0,2,10,1,2.,10,3,10,1,2.,0.0110750999301672,0.5113775134086609,0.2514517903327942,0,2,1,0,18,4,-1.,7,0,6,4,3.,-0.0367282517254353,0.7194662094116211,0.4849618971347809,0,2,4,3,6,1,-1.,6,3,2,1,3.,-2.8207109426148236e-004,0.3840261995792389,0.5394446253776550,0,2,4,18,13,2,-1.,4,19,13,1,2.,-2.7489690110087395e-003,0.5937088727951050,0.4569182097911835,0,2,2,10,3,6,-1.,2,12,3,2,3.,0.0100475195795298,0.5138576030731201,0.2802298069000244,0,3,14,12,6,8,-1.,17,12,3,4,2.,14,16,3,4,2.,-8.1497840583324432e-003,0.6090037226676941,0.4636121094226837,0,3,4,13,10,6,-1.,4,13,5,3,2.,9,16,5,3,2.,-6.8833888508379459e-003,0.3458611071109772,0.5254660248756409,0,2,14,12,1,2,-1.,14,13,1,1,2.,-1.4039360394235700e-005,0.5693104267120361,0.4082083106040955,0,2,8,13,4,3,-1.,8,14,4,1,3.,1.5498419525101781e-003,0.4350537061691284,0.5806517004966736,0,2,14,12,2,2,-1.,14,13,2,1,2.,-6.7841499112546444e-003,0.1468873023986816,0.5182775259017944,0,2,4,12,2,2,-1.,4,13,2,1,2.,2.1705629478674382e-004,0.5293524265289307,0.3456174135208130,0,2,8,12,9,2,-1.,8,13,9,1,2.,3.1198898795992136e-004,0.4652450978755951,0.5942413806915283,0,2,9,14,2,3,-1.,9,15,2,1,3.,5.4507530294358730e-003,0.4653508961200714,0.7024846076965332,0,2,11,10,3,6,-1.,11,13,3,3,2.,-2.5818689027801156e-004,0.5497295260429382,0.3768967092037201,0,2,5,6,9,12,-1.,5,12,9,6,2.,-0.0174425393342972,0.3919087946414948,0.5457497835159302,0,2,11,10,3,6,-1.,11,13,3,3,2.,-0.0453435294330120,0.1631357073783875,0.5154908895492554,0,2,6,10,3,6,-1.,6,13,3,3,2.,1.9190689781680703e-003,0.5145897865295410,0.2791895866394043,0,2,5,4,11,3,-1.,5,5,11,1,3.,-6.0177869163453579e-003,0.6517636179924011,0.4756332933902741,0,2,7,1,5,10,-1.,7,6,5,5,2.,-4.0720738470554352e-003,0.5514652729034424,0.4092685878276825,0,2,2,8,18,2,-1.,2,9,18,1,2.,3.9855059003457427e-004,0.3165240883827210,0.5285550951957703,0,2,7,17,5,3,-1.,7,18,5,1,3.,-6.5418570302426815e-003,0.6853377819061279,0.4652808904647827,0,2,5,9,12,1,-1.,9,9,4,1,3.,3.4845089539885521e-003,0.5484588146209717,0.4502759873867035,0,3,0,14,6,6,-1.,0,14,3,3,2.,3,17,3,3,2.,-0.0136967804282904,0.6395779848098755,0.4572555124759674,0,2,5,9,12,1,-1.,9,9,4,1,3.,-0.0173471402376890,0.2751072943210602,0.5181614756584168,0,2,3,9,12,1,-1.,7,9,4,1,3.,-4.0885428898036480e-003,0.3325636088848114,0.5194984078407288,0,2,14,10,6,7,-1.,14,10,3,7,2.,-9.4687901437282562e-003,0.5942280888557434,0.4851819872856140,0,2,1,0,16,2,-1.,1,1,16,1,2.,1.7084840219467878e-003,0.4167110919952393,0.5519806146621704,0,2,10,9,10,9,-1.,10,12,10,3,3.,9.4809094443917274e-003,0.5433894991874695,0.4208514988422394,0,2,0,1,10,2,-1.,5,1,5,2,2.,-4.7389650717377663e-003,0.6407189965248108,0.4560655057430267,0,2,17,3,2,3,-1.,17,4,2,1,3.,6.5761050209403038e-003,0.5214555263519287,0.2258227020502091,0,2,1,3,2,3,-1.,1,4,2,1,3.,-2.1690549328923225e-003,0.3151527941226959,0.5156704783439636,0,2,9,7,3,6,-1.,10,7,1,6,3.,0.0146601703017950,0.4870837032794952,0.6689941287040710,0,2,6,5,4,3,-1.,8,5,2,3,2.,1.7231999663636088e-004,0.3569748997688294,0.5251078009605408,0,2,7,5,6,6,-1.,9,5,2,6,3.,-0.0218037609010935,0.8825920820236206,0.4966329932212830,0,3,3,4,12,12,-1.,3,4,6,6,2.,9,10,6,6,2.,-0.0947361066937447,0.1446162015199661,0.5061113834381104,0,2,9,2,6,15,-1.,11,2,2,15,3.,5.5825551971793175e-003,0.5396478772163391,0.4238066077232361,0,2,2,2,6,17,-1.,4,2,2,17,3.,1.9517090404406190e-003,0.4170410931110382,0.5497786998748779,0,2,14,10,6,7,-1.,14,10,3,7,2.,0.0121499001979828,0.4698367118835449,0.5664274096488953,0,2,0,10,6,7,-1.,3,10,3,7,2.,-7.5169620104134083e-003,0.6267772912979126,0.4463135898113251,0,2,9,2,6,15,-1.,11,2,2,15,3.,-0.0716679096221924,0.3097011148929596,0.5221003293991089,0,2,5,2,6,15,-1.,7,2,2,15,3.,-0.0882924199104309,0.0811238884925842,0.5006365180015564,0,2,17,9,3,6,-1.,17,11,3,2,3.,0.0310630798339844,0.5155503749847412,0.1282255947589874,0,2,6,7,6,6,-1.,8,7,2,6,3.,0.0466218404471874,0.4699777960777283,0.7363960742950440,0,3,1,10,18,6,-1.,10,10,9,3,2.,1,13,9,3,2.,-0.0121894897893071,0.3920530080795288,0.5518996715545654,0,2,0,9,10,9,-1.,0,12,10,3,3.,0.0130161102861166,0.5260658264160156,0.3685136139392853,0,2,8,15,4,3,-1.,8,16,4,1,3.,-3.4952899441123009e-003,0.6339294910430908,0.4716280996799469,0,2,5,12,3,4,-1.,5,14,3,2,2.,-4.4015039748046547e-005,0.5333027243614197,0.3776184916496277,0,2,3,3,16,12,-1.,3,9,16,6,2.,-0.1096649020910263,0.1765342056751251,0.5198346972465515,0,3,1,1,12,12,-1.,1,1,6,6,2.,7,7,6,6,2.,-9.0279558207839727e-004,0.5324159860610962,0.3838908076286316,0,3,10,4,2,4,-1.,11,4,1,2,2.,10,6,1,2,2.,7.1126641705632210e-004,0.4647929966449738,0.5755224227905273,0,3,0,9,10,2,-1.,0,9,5,1,2.,5,10,5,1,2.,-3.1250279862433672e-003,0.3236708939075470,0.5166770815849304,0,2,9,11,3,3,-1.,9,12,3,1,3.,2.4144679773598909e-003,0.4787439107894898,0.6459717750549316,0,2,3,12,9,2,-1.,3,13,9,1,2.,4.4391240226104856e-004,0.4409308135509491,0.6010255813598633,0,2,9,9,2,2,-1.,9,10,2,1,2.,-2.2611189342569560e-004,0.4038113951683044,0.5493255853652954,66.6691207885742190,135,0,2,3,4,13,6,-1.,3,6,13,2,3.,-0.0469012893736362,0.6600171923637390,0.3743801116943359,0,3,9,7,6,4,-1.,12,7,3,2,2.,9,9,3,2,2.,-1.4568349579349160e-003,0.5783991217613220,0.3437797129154205,0,2,1,0,6,8,-1.,4,0,3,8,2.,5.5598369799554348e-003,0.3622266948223114,0.5908216238021851,0,2,9,5,2,12,-1.,9,11,2,6,2.,7.3170487303286791e-004,0.5500419139862061,0.2873558104038239,0,2,4,4,3,10,-1.,4,9,3,5,2.,1.3318009441718459e-003,0.2673169970512390,0.5431019067764282,0,2,6,17,8,3,-1.,6,18,8,1,3.,2.4347059661522508e-004,0.3855027854442596,0.5741388797760010,0,2,0,5,10,6,-1.,0,7,10,2,3.,-3.0512469820678234e-003,0.5503209829330444,0.3462845087051392,0,2,13,2,3,2,-1.,13,3,3,1,2.,-6.8657199153676629e-004,0.3291221857070923,0.5429509282112122,0,2,7,5,4,5,-1.,9,5,2,5,2.,1.4668200165033340e-003,0.3588382005691528,0.5351811051368713,0,2,12,14,3,6,-1.,12,16,3,2,3.,3.2021870720200241e-004,0.4296841919422150,0.5700234174728394,0,2,1,11,8,2,-1.,1,12,8,1,2.,7.4122188379988074e-004,0.5282164812088013,0.3366870880126953,0,2,7,13,6,3,-1.,7,14,6,1,3.,3.8330298848450184e-003,0.4559567868709564,0.6257336139678955,0,2,0,5,3,6,-1.,0,7,3,2,3.,-0.0154564399272203,0.2350116968154907,0.5129452943801880,0,2,13,2,3,2,-1.,13,3,3,1,2.,2.6796779129654169e-003,0.5329415202140808,0.4155062139034271,0,3,4,14,4,6,-1.,4,14,2,3,2.,6,17,2,3,2.,2.8296569362282753e-003,0.4273087978363037,0.5804538130760193,0,2,13,2,3,2,-1.,13,3,3,1,2.,-3.9444249123334885e-003,0.2912611961364746,0.5202686190605164,0,2,8,2,4,12,-1.,8,6,4,4,3.,2.7179559692740440e-003,0.5307688117027283,0.3585677146911621,0,3,14,0,6,8,-1.,17,0,3,4,2.,14,4,3,4,2.,5.9077627956867218e-003,0.4703775048255920,0.5941585898399353,0,2,7,17,3,2,-1.,8,17,1,2,3.,-4.2240349575877190e-003,0.2141567021608353,0.5088796019554138,0,2,8,12,4,2,-1.,8,13,4,1,2.,4.0725888684391975e-003,0.4766413867473602,0.6841061115264893,0,3,6,0,8,12,-1.,6,0,4,6,2.,10,6,4,6,2.,0.0101495301350951,0.5360798835754395,0.3748497068881989,0,3,14,0,2,10,-1.,15,0,1,5,2.,14,5,1,5,2.,-1.8864999583456665e-004,0.5720130205154419,0.3853805065155029,0,3,5,3,8,6,-1.,5,3,4,3,2.,9,6,4,3,2.,-4.8864358104765415e-003,0.3693122863769531,0.5340958833694458,0,3,14,0,6,10,-1.,17,0,3,5,2.,14,5,3,5,2.,0.0261584799736738,0.4962374866008759,0.6059989929199219,0,2,9,14,1,2,-1.,9,15,1,1,2.,4.8560759751126170e-004,0.4438945949077606,0.6012468934059143,0,2,15,10,4,3,-1.,15,11,4,1,3.,0.0112687097862363,0.5244250297546387,0.1840388029813767,0,2,8,14,2,3,-1.,8,15,2,1,3.,-2.8114619199186563e-003,0.6060283780097961,0.4409897029399872,0,3,3,13,14,4,-1.,10,13,7,2,2.,3,15,7,2,2.,-5.6112729944288731e-003,0.3891170918941498,0.5589237213134766,0,2,1,10,4,3,-1.,1,11,4,1,3.,8.5680093616247177e-003,0.5069345831871033,0.2062619030475617,0,2,9,11,6,1,-1.,11,11,2,1,3.,-3.8172779022715986e-004,0.5882201790809631,0.4192610979080200,0,2,5,11,6,1,-1.,7,11,2,1,3.,-1.7680290329735726e-004,0.5533605813980103,0.4003368914127350,0,2,3,5,16,15,-1.,3,10,16,5,3.,6.5112537704408169e-003,0.3310146927833557,0.5444191098213196,0,2,6,12,4,2,-1.,8,12,2,2,2.,-6.5948683186434209e-005,0.5433831810951233,0.3944905996322632,0,3,4,4,12,10,-1.,10,4,6,5,2.,4,9,6,5,2.,6.9939051754772663e-003,0.5600358247756958,0.4192714095115662,0,2,8,6,3,4,-1.,9,6,1,4,3.,-4.6744439750909805e-003,0.6685466766357422,0.4604960978031158,0,3,8,12,4,8,-1.,10,12,2,4,2.,8,16,2,4,2.,0.0115898502990603,0.5357121229171753,0.2926830053329468,0,2,8,14,4,3,-1.,8,15,4,1,3.,0.0130078401416540,0.4679817855358124,0.7307463288307190,0,2,12,2,3,2,-1.,13,2,1,2,3.,-1.1008579749614000e-003,0.3937501013278961,0.5415065288543701,0,2,8,15,3,2,-1.,8,16,3,1,2.,6.0472649056464434e-004,0.4242376089096069,0.5604041218757629,0,2,6,0,9,14,-1.,9,0,3,14,3.,-0.0144948400557041,0.3631210029125214,0.5293182730674744,0,2,9,6,2,3,-1.,10,6,1,3,2.,-5.3056948818266392e-003,0.6860452294349670,0.4621821045875549,0,2,10,8,2,3,-1.,10,9,2,1,3.,-8.1829127157106996e-004,0.3944096863269806,0.5420439243316650,0,2,0,9,4,6,-1.,0,11,4,2,3.,-0.0190775208175182,0.1962621957063675,0.5037891864776611,0,2,6,0,8,2,-1.,6,1,8,1,2.,3.5549470339901745e-004,0.4086259007453919,0.5613973140716553,0,2,6,14,7,3,-1.,6,15,7,1,3.,1.9679730758070946e-003,0.4489121139049530,0.5926123261451721,0,2,8,10,8,9,-1.,8,13,8,3,3.,6.9189141504466534e-003,0.5335925817489624,0.3728385865688324,0,2,5,2,3,2,-1.,6,2,1,2,3.,2.9872779268771410e-003,0.5111321210861206,0.2975643873214722,0,3,14,1,6,8,-1.,17,1,3,4,2.,14,5,3,4,2.,-6.2264618463814259e-003,0.5541489720344544,0.4824537932872772,0,3,0,1,6,8,-1.,0,1,3,4,2.,3,5,3,4,2.,0.0133533002808690,0.4586423933506012,0.6414797902107239,0,3,1,2,18,6,-1.,10,2,9,3,2.,1,5,9,3,2.,0.0335052385926247,0.5392425060272217,0.3429994881153107,0,2,9,3,2,1,-1.,10,3,1,1,2.,-2.5294460356235504e-003,0.1703713983297348,0.5013315081596375,0,3,13,2,4,6,-1.,15,2,2,3,2.,13,5,2,3,2.,-1.2801629491150379e-003,0.5305461883544922,0.4697405099868774,0,2,5,4,3,3,-1.,5,5,3,1,3.,7.0687388069927692e-003,0.4615545868873596,0.6436504721641541,0,2,13,5,1,3,-1.,13,6,1,1,3.,9.6880499040707946e-004,0.4833599030971527,0.6043894290924072,0,2,2,16,5,3,-1.,2,17,5,1,3.,3.9647659286856651e-003,0.5187637209892273,0.3231816887855530,0,3,13,2,4,6,-1.,15,2,2,3,2.,13,5,2,3,2.,-0.0220577307045460,0.4079256951808929,0.5200980901718140,0,3,3,2,4,6,-1.,3,2,2,3,2.,5,5,2,3,2.,-6.6906312713399529e-004,0.5331609249114990,0.3815600872039795,0,2,13,5,1,2,-1.,13,6,1,1,2.,-6.7009328631684184e-004,0.5655422210693359,0.4688901901245117,0,2,5,5,2,2,-1.,5,6,2,1,2.,7.4284552829340100e-004,0.4534381031990051,0.6287400126457214,0,2,13,9,2,2,-1.,13,9,1,2,2.,2.2227810695767403e-003,0.5350633263587952,0.3303655982017517,0,2,5,9,2,2,-1.,6,9,1,2,2.,-5.4130521602928638e-003,0.1113687008619309,0.5005434751510620,0,2,13,17,3,2,-1.,13,18,3,1,2.,-1.4520040167553816e-005,0.5628737807273865,0.4325133860111237,0,3,6,16,4,4,-1.,6,16,2,2,2.,8,18,2,2,2.,2.3369169502984732e-004,0.4165835082530975,0.5447791218757629,0,2,9,16,2,3,-1.,9,17,2,1,3.,4.2894547805190086e-003,0.4860391020774841,0.6778649091720581,0,2,0,13,9,6,-1.,0,15,9,2,3.,5.9103150852024555e-003,0.5262305140495300,0.3612113893032074,0,2,9,14,2,6,-1.,9,17,2,3,2.,0.0129005396738648,0.5319377183914185,0.3250288069248200,0,2,9,15,2,3,-1.,9,16,2,1,3.,4.6982979401946068e-003,0.4618245065212250,0.6665925979614258,0,2,1,10,18,6,-1.,1,12,18,2,3.,0.0104398597031832,0.5505670905113220,0.3883604109287262,0,2,8,11,4,2,-1.,8,12,4,1,2.,3.0443191062659025e-003,0.4697853028774262,0.7301844954490662,0,2,7,9,6,2,-1.,7,10,6,1,2.,-6.1593751888722181e-004,0.3830839097499847,0.5464984178543091,0,2,8,8,2,3,-1.,8,9,2,1,3.,-3.4247159492224455e-003,0.2566300034523010,0.5089530944824219,0,2,17,5,3,4,-1.,18,5,1,4,3.,-9.3538565561175346e-003,0.6469966173171997,0.4940795898437500,0,2,1,19,18,1,-1.,7,19,6,1,3.,0.0523389987647533,0.4745982885360718,0.7878770828247070,0,2,9,0,3,2,-1.,10,0,1,2,3.,3.5765620414167643e-003,0.5306664705276489,0.2748498022556305,0,2,1,8,1,6,-1.,1,10,1,2,3.,7.1555317845195532e-004,0.5413125753402710,0.4041908979415894,0,2,12,17,8,3,-1.,12,17,4,3,2.,-0.0105166798457503,0.6158512234687805,0.4815283119678497,0,2,0,5,3,4,-1.,1,5,1,4,3.,7.7347927726805210e-003,0.4695805907249451,0.7028980851173401,0,2,9,7,2,3,-1.,9,8,2,1,3.,-4.3226778507232666e-003,0.2849566042423248,0.5304684042930603,0,3,7,11,2,2,-1.,7,11,1,1,2.,8,12,1,1,2.,-2.5534399319440126e-003,0.7056984901428223,0.4688892066478729,0,2,11,3,2,5,-1.,11,3,1,5,2.,1.0268510231981054e-004,0.3902932107448578,0.5573464035987854,0,2,7,3,2,5,-1.,8,3,1,5,2.,7.1395188570022583e-006,0.3684231936931610,0.5263987779617310,0,2,15,13,2,3,-1.,15,14,2,1,3.,-1.6711989883333445e-003,0.3849175870418549,0.5387271046638489,0,2,5,6,2,3,-1.,5,7,2,1,3.,4.9260449595749378e-003,0.4729771912097931,0.7447251081466675,0,2,4,19,15,1,-1.,9,19,5,1,3.,4.3908702209591866e-003,0.4809181094169617,0.5591921806335449,0,2,1,19,15,1,-1.,6,19,5,1,3.,-0.0177936293184757,0.6903678178787231,0.4676927030086517,0,2,15,13,2,3,-1.,15,14,2,1,3.,2.0469669252634048e-003,0.5370690226554871,0.3308162093162537,0,2,5,0,4,15,-1.,7,0,2,15,2.,0.0298914890736341,0.5139865279197693,0.3309059143066406,0,2,9,6,2,5,-1.,9,6,1,5,2.,1.5494900289922953e-003,0.4660237133502960,0.6078342795372009,0,2,9,5,2,7,-1.,10,5,1,7,2.,1.4956969534978271e-003,0.4404835999011993,0.5863919854164124,0,2,16,11,3,3,-1.,16,12,3,1,3.,9.5885928021743894e-004,0.5435971021652222,0.4208523035049439,0,2,1,11,3,3,-1.,1,12,3,1,3.,4.9643701640889049e-004,0.5370578169822693,0.4000622034072876,0,2,6,6,8,3,-1.,6,7,8,1,3.,-2.7280810754746199e-003,0.5659412741661072,0.4259642958641052,0,2,0,15,6,2,-1.,0,16,6,1,2.,2.3026480339467525e-003,0.5161657929420471,0.3350869119167328,0,2,1,0,18,6,-1.,7,0,6,6,3.,0.2515163123607636,0.4869661927223206,0.7147309780120850,0,2,6,0,3,4,-1.,7,0,1,4,3.,-4.6328022144734859e-003,0.2727448940277100,0.5083789825439453,0,3,14,10,4,10,-1.,16,10,2,5,2.,14,15,2,5,2.,-0.0404344908893108,0.6851438879966736,0.5021767020225525,0,2,3,2,3,2,-1.,4,2,1,2,3.,1.4972220014897175e-005,0.4284465014934540,0.5522555112838745,0,2,11,2,2,2,-1.,11,3,2,1,2.,-2.4050309730228037e-004,0.4226118922233582,0.5390074849128723,0,3,2,10,4,10,-1.,2,10,2,5,2.,4,15,2,5,2.,0.0236578397452831,0.4744631946086884,0.7504366040229797,0,3,0,13,20,6,-1.,10,13,10,3,2.,0,16,10,3,2.,-8.1449104472994804e-003,0.4245058894157410,0.5538362860679627,0,2,0,5,2,15,-1.,1,5,1,15,2.,-3.6992130335420370e-003,0.5952357053756714,0.4529713094234467,0,3,1,7,18,4,-1.,10,7,9,2,2.,1,9,9,2,2.,-6.7718601785600185e-003,0.4137794077396393,0.5473399758338928,0,2,0,0,2,17,-1.,1,0,1,17,2.,4.2669530957937241e-003,0.4484114944934845,0.5797994136810303,0,3,2,6,16,6,-1.,10,6,8,3,2.,2,9,8,3,2.,1.7791989957913756e-003,0.5624858736991882,0.4432444870471954,0,2,8,14,1,3,-1.,8,15,1,1,3.,1.6774770338088274e-003,0.4637751877307892,0.6364241838455200,0,2,8,15,4,2,-1.,8,16,4,1,2.,1.1732629500329494e-003,0.4544503092765808,0.5914415717124939,0,3,5,2,8,2,-1.,5,2,4,1,2.,9,3,4,1,2.,8.6998171173036098e-004,0.5334752798080444,0.3885917961597443,0,2,6,11,8,6,-1.,6,14,8,3,2.,7.6378340600058436e-004,0.5398585200309753,0.3744941949844360,0,2,9,13,2,2,-1.,9,14,2,1,2.,1.5684569370932877e-004,0.4317873120307922,0.5614616274833679,0,2,18,4,2,6,-1.,18,6,2,2,3.,-0.0215113703161478,0.1785925030708313,0.5185542702674866,0,2,9,12,2,2,-1.,9,13,2,1,2.,1.3081369979772717e-004,0.4342499077320099,0.5682849884033203,0,2,18,4,2,6,-1.,18,6,2,2,3.,0.0219920407980680,0.5161716938018799,0.2379394024610519,0,2,9,13,1,3,-1.,9,14,1,1,3.,-8.0136500764638186e-004,0.5986763238906860,0.4466426968574524,0,2,18,4,2,6,-1.,18,6,2,2,3.,-8.2736099138855934e-003,0.4108217954635620,0.5251057147979736,0,2,0,4,2,6,-1.,0,6,2,2,3.,3.6831789184361696e-003,0.5173814296722412,0.3397518098354340,0,2,9,12,3,3,-1.,9,13,3,1,3.,-7.9525681212544441e-003,0.6888983249664307,0.4845924079418182,0,2,3,13,2,3,-1.,3,14,2,1,3.,1.5382299898192286e-003,0.5178567171096802,0.3454113900661469,0,2,13,13,4,3,-1.,13,14,4,1,3.,-0.0140435304492712,0.1678421050310135,0.5188667774200440,0,2,5,4,3,3,-1.,5,5,3,1,3.,1.4315890148282051e-003,0.4368256926536560,0.5655773878097534,0,2,5,2,10,6,-1.,5,4,10,2,3.,-0.0340142287313938,0.7802296280860901,0.4959217011928558,0,2,3,13,4,3,-1.,3,14,4,1,3.,-0.0120272999629378,0.1585101038217545,0.5032231807708740,0,2,3,7,15,5,-1.,8,7,5,5,3.,0.1331661939620972,0.5163304805755615,0.2755128145217896,0,2,3,7,12,2,-1.,7,7,4,2,3.,-1.5221949433907866e-003,0.3728317916393280,0.5214552283287048,0,2,10,3,3,9,-1.,11,3,1,9,3.,-9.3929271679371595e-004,0.5838379263877869,0.4511165022850037,0,2,8,6,4,6,-1.,10,6,2,6,2.,0.0277197398245335,0.4728286862373352,0.7331544756889343,0,2,9,7,4,3,-1.,9,8,4,1,3.,3.1030150130391121e-003,0.5302202105522156,0.4101563096046448,0,2,0,9,4,9,-1.,2,9,2,9,2.,0.0778612196445465,0.4998334050178528,0.1272961944341660,0,2,9,13,3,5,-1.,10,13,1,5,3.,-0.0158549398183823,0.0508333593606949,0.5165656208992004,0,2,7,7,6,3,-1.,9,7,2,3,3.,-4.9725300632417202e-003,0.6798133850097656,0.4684231877326965,0,2,9,7,3,5,-1.,10,7,1,5,3.,-9.7676506265997887e-004,0.6010771989822388,0.4788931906223297,0,2,5,7,8,2,-1.,9,7,4,2,2.,-2.4647710379213095e-003,0.3393397927284241,0.5220503807067871,0,2,5,9,12,2,-1.,9,9,4,2,3.,-6.7937700077891350e-003,0.4365136921405792,0.5239663124084473,0,2,5,6,10,3,-1.,10,6,5,3,2.,0.0326080210506916,0.5052723884582520,0.2425214946269989,0,2,10,12,3,1,-1.,11,12,1,1,3.,-5.8514421107247472e-004,0.5733973979949951,0.4758574068546295,0,2,0,1,11,15,-1.,0,6,11,5,3.,-0.0296326000243425,0.3892289102077484,0.5263597965240479,67.6989212036132810,137,0,2,1,0,18,6,-1.,7,0,6,6,3.,0.0465508513152599,0.3276950120925903,0.6240522861480713,0,2,7,7,6,1,-1.,9,7,2,1,3.,7.9537127166986465e-003,0.4256485104560852,0.6942939162254334,0,3,5,16,6,4,-1.,5,16,3,2,2.,8,18,3,2,2.,6.8221561377868056e-004,0.3711487054824829,0.5900732874870300,0,2,6,5,9,8,-1.,6,9,9,4,2.,-1.9348249770700932e-004,0.2041133940219879,0.5300545096397400,0,2,5,10,2,6,-1.,5,13,2,3,2.,-2.6710508973337710e-004,0.5416126251220703,0.3103179037570953,0,3,7,6,8,10,-1.,11,6,4,5,2.,7,11,4,5,2.,2.7818060480058193e-003,0.5277832746505737,0.3467069864273071,0,3,5,6,8,10,-1.,5,6,4,5,2.,9,11,4,5,2.,-4.6779078547842801e-004,0.5308231115341187,0.3294492065906525,0,2,9,5,2,2,-1.,9,6,2,1,2.,-3.0335160772665404e-005,0.5773872733116150,0.3852097094058991,0,2,5,12,8,2,-1.,5,13,8,1,2.,7.8038009814918041e-004,0.4317438900470734,0.6150057911872864,0,2,10,2,8,2,-1.,10,3,8,1,2.,-4.2553851380944252e-003,0.2933903932571411,0.5324292778968811,0,3,4,0,2,10,-1.,4,0,1,5,2.,5,5,1,5,2.,-2.4735610350035131e-004,0.5468844771385193,0.3843030035495758,0,2,9,10,2,2,-1.,9,11,2,1,2.,-1.4724259381182492e-004,0.4281542897224426,0.5755587220191956,0,2,2,8,15,3,-1.,2,9,15,1,3.,1.1864770203828812e-003,0.3747301101684570,0.5471466183662415,0,2,8,13,4,3,-1.,8,14,4,1,3.,2.3936580400913954e-003,0.4537783861160278,0.6111528873443604,0,2,7,2,3,2,-1.,8,2,1,2,3.,-1.5390539774671197e-003,0.2971341907978058,0.5189538002014160,0,2,7,13,6,3,-1.,7,14,6,1,3.,-7.1968790143728256e-003,0.6699066758155823,0.4726476967334747,0,2,9,9,2,2,-1.,9,10,2,1,2.,-4.1499789222143590e-004,0.3384954035282135,0.5260317921638489,0,2,17,2,3,6,-1.,17,4,3,2,3.,4.4359830208122730e-003,0.5399122238159180,0.3920140862464905,0,2,1,5,3,4,-1.,2,5,1,4,3.,2.6606200262904167e-003,0.4482578039169312,0.6119617819786072,0,2,14,8,4,6,-1.,14,10,4,2,3.,-1.5287200221791863e-003,0.3711237907409668,0.5340266227722168,0,2,1,4,3,8,-1.,2,4,1,8,3.,-4.7397250309586525e-003,0.6031088232994080,0.4455145001411438,0,2,8,13,4,6,-1.,8,16,4,3,2.,-0.0148291299119592,0.2838754057884216,0.5341861844062805,0,2,3,14,2,2,-1.,3,15,2,1,2.,9.2275557108223438e-004,0.5209547281265259,0.3361653983592987,0,2,14,8,4,6,-1.,14,10,4,2,3.,0.0835298076272011,0.5119969844818115,0.0811644494533539,0,2,2,8,4,6,-1.,2,10,4,2,3.,-7.5633148662745953e-004,0.3317120075225830,0.5189831256866455,0,2,10,14,1,6,-1.,10,17,1,3,2.,9.8403859883546829e-003,0.5247598290443420,0.2334959059953690,0,2,7,5,3,6,-1.,8,5,1,6,3.,-1.5953830443322659e-003,0.5750094056129456,0.4295622110366821,0,3,11,2,2,6,-1.,12,2,1,3,2.,11,5,1,3,2.,3.4766020689858124e-005,0.4342445135116577,0.5564029216766357,0,2,6,6,6,5,-1.,8,6,2,5,3.,0.0298629105091095,0.4579147100448608,0.6579188108444214,0,2,17,1,3,6,-1.,17,3,3,2,3.,0.0113255903124809,0.5274311900138855,0.3673888146877289,0,2,8,7,3,5,-1.,9,7,1,5,3.,-8.7828645482659340e-003,0.7100368738174439,0.4642167091369629,0,2,9,18,3,2,-1.,10,18,1,2,3.,4.3639959767460823e-003,0.5279216170310974,0.2705877125263214,0,2,8,18,3,2,-1.,9,18,1,2,3.,4.1804728098213673e-003,0.5072525143623352,0.2449083030223846,0,2,12,3,5,2,-1.,12,4,5,1,2.,-4.5668511302210391e-004,0.4283105134963989,0.5548691153526306,0,2,7,1,5,12,-1.,7,7,5,6,2.,-3.7140368949621916e-003,0.5519387722015381,0.4103653132915497,0,2,1,0,18,4,-1.,7,0,6,4,3.,-0.0253042895346880,0.6867002248764038,0.4869889020919800,0,2,4,2,2,2,-1.,4,3,2,1,2.,-3.4454080741852522e-004,0.3728874027729034,0.5287693142890930,0,3,11,14,4,2,-1.,13,14,2,1,2.,11,15,2,1,2.,-8.3935231668874621e-004,0.6060152053833008,0.4616062045097351,0,2,0,2,3,6,-1.,0,4,3,2,3.,0.0172800496220589,0.5049635767936707,0.1819823980331421,0,2,9,7,2,3,-1.,9,8,2,1,3.,-6.3595077954232693e-003,0.1631239950656891,0.5232778787612915,0,2,5,5,1,3,-1.,5,6,1,1,3.,1.0298109846189618e-003,0.4463278055191040,0.6176549196243286,0,2,10,10,6,1,-1.,10,10,3,1,2.,1.0117109632119536e-003,0.5473384857177734,0.4300698935985565,0,2,4,10,6,1,-1.,7,10,3,1,2.,-0.0103088002651930,0.1166985034942627,0.5000867247581482,0,2,9,17,3,3,-1.,9,18,3,1,3.,5.4682018235325813e-003,0.4769287109375000,0.6719213724136353,0,2,4,14,1,3,-1.,4,15,1,1,3.,-9.1696460731327534e-004,0.3471089899539948,0.5178164839744568,0,2,12,5,3,3,-1.,12,6,3,1,3.,2.3922820109874010e-003,0.4785236120223999,0.6216310858726502,0,2,4,5,12,3,-1.,4,6,12,1,3.,-7.5573818758130074e-003,0.5814796090126038,0.4410085082054138,0,2,9,8,2,3,-1.,9,9,2,1,3.,-7.7024032361805439e-004,0.3878000080585480,0.5465722084045410,0,2,4,9,3,3,-1.,5,9,1,3,3.,-8.7125990539789200e-003,0.1660051047801971,0.4995836019515991,0,2,6,0,9,17,-1.,9,0,3,17,3.,-0.0103063201531768,0.4093391001224518,0.5274233818054199,0,2,9,12,1,3,-1.,9,13,1,1,3.,-2.0940979011356831e-003,0.6206194758415222,0.4572280049324036,0,2,9,5,2,15,-1.,9,10,2,5,3.,6.8099051713943481e-003,0.5567759275436401,0.4155600070953369,0,2,8,14,2,3,-1.,8,15,2,1,3.,-1.0746059706434608e-003,0.5638927817344666,0.4353024959564209,0,2,10,14,1,3,-1.,10,15,1,1,3.,2.1550289820879698e-003,0.4826265871524811,0.6749758124351502,0,2,7,1,6,5,-1.,9,1,2,5,3.,0.0317423194646835,0.5048379898071289,0.1883248984813690,0,2,0,0,20,2,-1.,0,0,10,2,2.,-0.0783827230334282,0.2369548976421356,0.5260158181190491,0,2,2,13,5,3,-1.,2,14,5,1,3.,5.7415119372308254e-003,0.5048828721046448,0.2776469886302948,0,2,9,11,2,3,-1.,9,12,2,1,3.,-2.9014600440859795e-003,0.6238604784011841,0.4693317115306854,0,2,2,5,9,15,-1.,2,10,9,5,3.,-2.6427931152284145e-003,0.3314141929149628,0.5169777274131775,0,3,5,0,12,10,-1.,11,0,6,5,2.,5,5,6,5,2.,-0.1094966009259224,0.2380045056343079,0.5183441042900085,0,2,5,1,2,3,-1.,6,1,1,3,2.,7.4075913289561868e-005,0.4069635868072510,0.5362150073051453,0,2,10,7,6,1,-1.,12,7,2,1,3.,-5.0593802006915212e-004,0.5506706237792969,0.4374594092369080,0,3,3,1,2,10,-1.,3,1,1,5,2.,4,6,1,5,2.,-8.2131777890026569e-004,0.5525709986686707,0.4209375977516174,0,2,13,7,2,1,-1.,13,7,1,1,2.,-6.0276539443293586e-005,0.5455474853515625,0.4748266041278839,0,2,4,13,4,6,-1.,4,15,4,2,3.,6.8065142259001732e-003,0.5157995820045471,0.3424577116966248,0,2,13,7,2,1,-1.,13,7,1,1,2.,1.7202789895236492e-003,0.5013207793235779,0.6331263780593872,0,2,5,7,2,1,-1.,6,7,1,1,2.,-1.3016929733566940e-004,0.5539718270301819,0.4226869940757752,0,3,2,12,18,4,-1.,11,12,9,2,2.,2,14,9,2,2.,-4.8016388900578022e-003,0.4425095021724701,0.5430780053138733,0,3,5,7,2,2,-1.,5,7,1,1,2.,6,8,1,1,2.,-2.5399310979992151e-003,0.7145782113075256,0.4697605073451996,0,2,16,3,4,2,-1.,16,4,4,1,2.,-1.4278929447755218e-003,0.4070445001125336,0.5399605035781860,0,3,0,2,2,18,-1.,0,2,1,9,2.,1,11,1,9,2.,-0.0251425504684448,0.7884690761566162,0.4747352004051209,0,3,1,2,18,4,-1.,10,2,9,2,2.,1,4,9,2,2.,-3.8899609353393316e-003,0.4296191930770874,0.5577110052108765,0,2,9,14,1,3,-1.,9,15,1,1,3.,4.3947459198534489e-003,0.4693162143230438,0.7023944258689880,0,3,2,12,18,4,-1.,11,12,9,2,2.,2,14,9,2,2.,0.0246784202754498,0.5242322087287903,0.3812510073184967,0,3,0,12,18,4,-1.,0,12,9,2,2.,9,14,9,2,2.,0.0380476787686348,0.5011739730834961,0.1687828004360199,0,2,11,4,5,3,-1.,11,5,5,1,3.,7.9424865543842316e-003,0.4828582108020783,0.6369568109512329,0,2,6,4,7,3,-1.,6,5,7,1,3.,-1.5110049862414598e-003,0.5906485915184021,0.4487667977809906,0,2,13,17,3,3,-1.,13,18,3,1,3.,6.4201741479337215e-003,0.5241097807884216,0.2990570068359375,0,2,8,1,3,4,-1.,9,1,1,4,3.,-2.9802159406244755e-003,0.3041465878486633,0.5078489780426025,0,2,11,4,2,4,-1.,11,4,1,4,2.,-7.4580078944563866e-004,0.4128139019012451,0.5256826281547546,0,2,0,17,9,3,-1.,3,17,3,3,3.,-0.0104709500446916,0.5808395147323608,0.4494296014308929,0,3,11,0,2,8,-1.,12,0,1,4,2.,11,4,1,4,2.,9.3369204550981522e-003,0.5246552824974060,0.2658948898315430,0,3,0,8,6,12,-1.,0,8,3,6,2.,3,14,3,6,2.,0.0279369000345469,0.4674955010414124,0.7087256908416748,0,2,10,7,4,12,-1.,10,13,4,6,2.,7.4277678504586220e-003,0.5409486889839172,0.3758518099784851,0,2,5,3,8,14,-1.,5,10,8,7,2.,-0.0235845092684031,0.3758639991283417,0.5238550901412964,0,2,14,10,6,1,-1.,14,10,3,1,2.,1.1452640173956752e-003,0.4329578876495361,0.5804247260093689,0,2,0,4,10,4,-1.,0,6,10,2,2.,-4.3468660442158580e-004,0.5280618071556091,0.3873069882392883,0,2,10,0,5,8,-1.,10,4,5,4,2.,0.0106485402211547,0.4902113080024719,0.5681251883506775,0,3,8,1,4,8,-1.,8,1,2,4,2.,10,5,2,4,2.,-3.9418050437234342e-004,0.5570880174636841,0.4318251013755798,0,2,9,11,6,1,-1.,11,11,2,1,3.,-1.3270479394122958e-004,0.5658439993858337,0.4343554973602295,0,2,8,9,3,4,-1.,9,9,1,4,3.,-2.0125510636717081e-003,0.6056739091873169,0.4537523984909058,0,2,18,4,2,6,-1.,18,6,2,2,3.,2.4854319635778666e-003,0.5390477180480957,0.4138010144233704,0,2,8,8,3,4,-1.,9,8,1,4,3.,1.8237880431115627e-003,0.4354828894138336,0.5717188715934753,0,2,7,1,13,3,-1.,7,2,13,1,3.,-0.0166566595435143,0.3010913133621216,0.5216122865676880,0,2,7,13,6,1,-1.,9,13,2,1,3.,8.0349558265879750e-004,0.5300151109695435,0.3818396925926209,0,2,12,11,3,6,-1.,12,13,3,2,3.,3.4170378930866718e-003,0.5328028798103333,0.4241400063037872,0,2,5,11,6,1,-1.,7,11,2,1,3.,-3.6222729249857366e-004,0.5491728186607361,0.4186977148056030,0,3,1,4,18,10,-1.,10,4,9,5,2.,1,9,9,5,2.,-0.1163002029061317,0.1440722048282623,0.5226451158523560,0,2,8,6,4,9,-1.,8,9,4,3,3.,-0.0146950101479888,0.7747725248336792,0.4715717136859894,0,2,8,6,4,3,-1.,8,7,4,1,3.,2.1972130052745342e-003,0.5355433821678162,0.3315644860267639,0,2,8,7,3,3,-1.,9,7,1,3,3.,-4.6965209185145795e-004,0.5767235159873962,0.4458136856555939,0,2,14,15,4,3,-1.,14,16,4,1,3.,6.5144998952746391e-003,0.5215674042701721,0.3647888898849487,0,2,5,10,3,10,-1.,6,10,1,10,3.,0.0213000606745481,0.4994204938411713,0.1567950993776321,0,2,8,15,4,3,-1.,8,16,4,1,3.,3.1881409231573343e-003,0.4742200076580048,0.6287270188331604,0,2,0,8,1,6,-1.,0,10,1,2,3.,9.0019777417182922e-004,0.5347954034805298,0.3943752050399780,0,2,10,15,1,3,-1.,10,16,1,1,3.,-5.1772277802228928e-003,0.6727191805839539,0.5013138055801392,0,2,2,15,4,3,-1.,2,16,4,1,3.,-4.3764649890363216e-003,0.3106675148010254,0.5128793120384216,0,3,18,3,2,8,-1.,19,3,1,4,2.,18,7,1,4,2.,2.6299960445612669e-003,0.4886310100555420,0.5755215883255005,0,3,0,3,2,8,-1.,0,3,1,4,2.,1,7,1,4,2.,-2.0458688959479332e-003,0.6025794148445129,0.4558076858520508,0,3,3,7,14,10,-1.,10,7,7,5,2.,3,12,7,5,2.,0.0694827064871788,0.5240747928619385,0.2185259014368057,0,2,0,7,19,3,-1.,0,8,19,1,3.,0.0240489393472672,0.5011867284774780,0.2090622037649155,0,2,12,6,3,3,-1.,12,7,3,1,3.,3.1095340382307768e-003,0.4866712093353272,0.7108548283576965,0,2,0,6,1,3,-1.,0,7,1,1,3.,-1.2503260513767600e-003,0.3407891094684601,0.5156195163726807,0,2,12,6,3,3,-1.,12,7,3,1,3.,-1.0281190043315291e-003,0.5575572252273560,0.4439432024955750,0,2,5,6,3,3,-1.,5,7,3,1,3.,-8.8893622159957886e-003,0.6402000784873962,0.4620442092418671,0,2,8,2,4,2,-1.,8,3,4,1,2.,-6.1094801640138030e-004,0.3766441941261292,0.5448899865150452,0,2,6,3,4,12,-1.,8,3,2,12,2.,-5.7686357758939266e-003,0.3318648934364319,0.5133677124977112,0,2,13,6,2,3,-1.,13,7,2,1,3.,1.8506490159779787e-003,0.4903570115566254,0.6406934857368469,0,2,0,10,20,4,-1.,0,12,20,2,2.,-0.0997994691133499,0.1536051034927368,0.5015562176704407,0,2,2,0,17,14,-1.,2,7,17,7,2.,-0.3512834906578064,0.0588231310248375,0.5174378752708435,0,3,0,0,6,10,-1.,0,0,3,5,2.,3,5,3,5,2.,-0.0452445708215237,0.6961488723754883,0.4677872955799103,0,2,14,6,6,4,-1.,14,6,3,4,2.,0.0714815780520439,0.5167986154556274,0.1038092970848084,0,2,0,6,6,4,-1.,3,6,3,4,2.,2.1895780228078365e-003,0.4273078143596649,0.5532060861587524,0,2,13,2,7,2,-1.,13,3,7,1,2.,-5.9242651332169771e-004,0.4638943970203400,0.5276389122009277,0,2,0,2,7,2,-1.,0,3,7,1,2.,1.6788389766588807e-003,0.5301648974418640,0.3932034969329834,0,3,6,11,14,2,-1.,13,11,7,1,2.,6,12,7,1,2.,-2.2163488902151585e-003,0.5630694031715393,0.4757033884525299,0,3,8,5,2,2,-1.,8,5,1,1,2.,9,6,1,1,2.,1.1568699846975505e-004,0.4307535886764526,0.5535702705383301,0,2,13,9,2,3,-1.,13,9,1,3,2.,-7.2017288766801357e-003,0.1444882005453110,0.5193064212799072,0,2,1,1,3,12,-1.,2,1,1,12,3.,8.9081272017210722e-004,0.4384432137012482,0.5593621134757996,0,2,17,4,1,3,-1.,17,5,1,1,3.,1.9605009583756328e-004,0.5340415835380554,0.4705956876277924,0,2,2,4,1,3,-1.,2,5,1,1,3.,5.2022142335772514e-004,0.5213856101036072,0.3810079097747803,0,2,14,5,1,3,-1.,14,6,1,1,3.,9.4588572392240167e-004,0.4769414961338043,0.6130738854408264,0,2,7,16,2,3,-1.,7,17,2,1,3.,9.1698471806012094e-005,0.4245009124279022,0.5429363250732422,0,3,8,13,4,6,-1.,10,13,2,3,2.,8,16,2,3,2.,2.1833200007677078e-003,0.5457730889320374,0.4191075861454010,0,2,5,5,1,3,-1.,5,6,1,1,3.,-8.6039671441540122e-004,0.5764588713645935,0.4471659958362579,0,2,16,0,4,20,-1.,16,0,2,20,2.,-0.0132362395524979,0.6372823119163513,0.4695009887218475,0,3,5,1,2,6,-1.,5,1,1,3,2.,6,4,1,3,2.,4.3376701069064438e-004,0.5317873954772949,0.3945829868316650,69.2298736572265630,140,0,2,5,4,10,4,-1.,5,6,10,2,2.,-0.0248471498489380,0.6555516719818115,0.3873311877250671,0,2,15,2,4,12,-1.,15,2,2,12,2.,6.1348611488938332e-003,0.3748072087764740,0.5973997712135315,0,2,7,6,4,12,-1.,7,12,4,6,2.,6.4498498104512691e-003,0.5425491929054260,0.2548811137676239,0,2,14,5,1,8,-1.,14,9,1,4,2.,6.3491211039945483e-004,0.2462442070245743,0.5387253761291504,0,3,1,4,14,10,-1.,1,4,7,5,2.,8,9,7,5,2.,1.4023890253156424e-003,0.5594322085380554,0.3528657853603363,0,3,11,6,6,14,-1.,14,6,3,7,2.,11,13,3,7,2.,3.0044000595808029e-004,0.3958503901958466,0.5765938162803650,0,3,3,6,6,14,-1.,3,6,3,7,2.,6,13,3,7,2.,1.0042409849120304e-004,0.3698996901512146,0.5534998178482056,0,2,4,9,15,2,-1.,9,9,5,2,3.,-5.0841490738093853e-003,0.3711090981960297,0.5547800064086914,0,2,7,14,6,3,-1.,7,15,6,1,3.,-0.0195372607558966,0.7492755055427551,0.4579297006130219,0,3,6,3,14,4,-1.,13,3,7,2,2.,6,5,7,2,2.,-7.4532740654831287e-006,0.5649787187576294,0.3904069960117340,0,2,1,9,15,2,-1.,6,9,5,2,3.,-3.6079459823668003e-003,0.3381088078022003,0.5267801284790039,0,2,6,11,8,9,-1.,6,14,8,3,3.,2.0697501022368670e-003,0.5519291162490845,0.3714388906955719,0,2,7,4,3,8,-1.,8,4,1,8,3.,-4.6463840408250690e-004,0.5608214735984802,0.4113566875457764,0,2,14,6,2,6,-1.,14,9,2,3,2.,7.5490452582016587e-004,0.3559206128120422,0.5329356193542481,0,3,5,7,6,4,-1.,5,7,3,2,2.,8,9,3,2,2.,-9.8322238773107529e-004,0.5414795875549316,0.3763205111026764,0,2,1,1,18,19,-1.,7,1,6,19,3.,-0.0199406407773495,0.6347903013229370,0.4705299139022827,0,2,1,2,6,5,-1.,4,2,3,5,2.,3.7680300883948803e-003,0.3913489878177643,0.5563716292381287,0,2,12,17,6,2,-1.,12,18,6,1,2.,-9.4528505578637123e-003,0.2554892897605896,0.5215116739273071,0,2,2,17,6,2,-1.,2,18,6,1,2.,2.9560849070549011e-003,0.5174679160118103,0.3063920140266419,0,2,17,3,3,6,-1.,17,5,3,2,3.,9.1078737750649452e-003,0.5388448238372803,0.2885963022708893,0,2,8,17,3,3,-1.,8,18,3,1,3.,1.8219229532405734e-003,0.4336043000221252,0.5852196812629700,0,2,10,13,2,6,-1.,10,16,2,3,2.,0.0146887395530939,0.5287361741065979,0.2870005965232849,0,2,7,13,6,3,-1.,7,14,6,1,3.,-0.0143879903480411,0.7019448876380920,0.4647370874881744,0,2,17,3,3,6,-1.,17,5,3,2,3.,-0.0189866498112679,0.2986552119255066,0.5247011780738831,0,2,8,13,2,3,-1.,8,14,2,1,3.,1.1527639580890536e-003,0.4323473870754242,0.5931661725044251,0,2,9,3,6,2,-1.,11,3,2,2,3.,0.0109336702153087,0.5286864042282105,0.3130319118499756,0,2,0,3,3,6,-1.,0,5,3,2,3.,-0.0149327302351594,0.2658419013023377,0.5084077119827271,0,2,8,5,4,6,-1.,8,7,4,2,3.,-2.9970539617352188e-004,0.5463526844978333,0.3740724027156830,0,2,5,5,3,2,-1.,5,6,3,1,2.,4.1677621193230152e-003,0.4703496992588043,0.7435721755027771,0,2,10,1,3,4,-1.,11,1,1,4,3.,-6.3905320130288601e-003,0.2069258987903595,0.5280538201332092,0,2,1,2,5,9,-1.,1,5,5,3,3.,4.5029609464108944e-003,0.5182648897171021,0.3483543097972870,0,2,13,6,2,3,-1.,13,7,2,1,3.,-9.2040365561842918e-003,0.6803777217864990,0.4932360053062439,0,2,0,6,14,3,-1.,7,6,7,3,2.,0.0813272595405579,0.5058398842811585,0.2253051996231079,0,2,2,11,18,8,-1.,2,15,18,4,2.,-0.1507928073406220,0.2963424921035767,0.5264679789543152,0,2,5,6,2,3,-1.,5,7,2,1,3.,3.3179009333252907e-003,0.4655495882034302,0.7072932124137878,0,3,10,6,4,2,-1.,12,6,2,1,2.,10,7,2,1,2.,7.7402801252901554e-004,0.4780347943305969,0.5668237805366516,0,3,6,6,4,2,-1.,6,6,2,1,2.,8,7,2,1,2.,6.8199541419744492e-004,0.4286996126174927,0.5722156763076782,0,2,10,1,3,4,-1.,11,1,1,4,3.,5.3671570494771004e-003,0.5299307107925415,0.3114621937274933,0,2,7,1,2,7,-1.,8,1,1,7,2.,9.7018666565418243e-005,0.3674638867378235,0.5269461870193481,0,2,4,2,15,14,-1.,4,9,15,7,2.,-0.1253408938646317,0.2351492047309876,0.5245791077613831,0,2,8,7,3,2,-1.,9,7,1,2,3.,-5.2516269497573376e-003,0.7115936875343323,0.4693767130374908,0,3,2,3,18,4,-1.,11,3,9,2,2.,2,5,9,2,2.,-7.8342109918594360e-003,0.4462651014328003,0.5409085750579834,0,2,9,7,2,2,-1.,10,7,1,2,2.,-1.1310069821774960e-003,0.5945618748664856,0.4417662024497986,0,2,13,9,2,3,-1.,13,9,1,3,2.,1.7601120052859187e-003,0.5353249907493591,0.3973453044891357,0,2,5,2,6,2,-1.,7,2,2,2,3.,-8.1581249833106995e-004,0.3760268092155457,0.5264726877212524,0,2,9,5,2,7,-1.,9,5,1,7,2.,-3.8687589112669230e-003,0.6309912800788879,0.4749819934368134,0,2,5,9,2,3,-1.,6,9,1,3,2.,1.5207129763439298e-003,0.5230181813240051,0.3361223936080933,0,2,6,0,14,18,-1.,6,9,14,9,2.,0.5458673834800720,0.5167139768600464,0.1172635033726692,0,2,2,16,6,3,-1.,2,17,6,1,3.,0.0156501904129982,0.4979439079761505,0.1393294930458069,0,2,9,7,3,6,-1.,10,7,1,6,3.,-0.0117318602278829,0.7129650712013245,0.4921196103096008,0,2,7,8,4,3,-1.,7,9,4,1,3.,-6.1765122227370739e-003,0.2288102954626083,0.5049701929092407,0,2,7,12,6,3,-1.,7,13,6,1,3.,2.2457661107182503e-003,0.4632433950901032,0.6048725843429565,0,2,9,12,2,3,-1.,9,13,2,1,3.,-5.1915869116783142e-003,0.6467421054840088,0.4602192938327789,0,2,7,12,6,2,-1.,9,12,2,2,3.,-0.0238278806209564,0.1482000946998596,0.5226079225540161,0,2,5,11,4,6,-1.,5,14,4,3,2.,1.0284580057486892e-003,0.5135489106178284,0.3375957012176514,0,2,11,12,7,2,-1.,11,13,7,1,2.,-0.0100788502022624,0.2740561068058014,0.5303567051887512,0,3,6,10,8,6,-1.,6,10,4,3,2.,10,13,4,3,2.,2.6168930344283581e-003,0.5332670807838440,0.3972454071044922,0,2,11,10,3,4,-1.,11,12,3,2,2.,5.4385367548093200e-004,0.5365604162216187,0.4063411951065064,0,2,9,16,2,3,-1.,9,17,2,1,3.,5.3510512225329876e-003,0.4653759002685547,0.6889045834541321,0,2,13,3,1,9,-1.,13,6,1,3,3.,-1.5274790348485112e-003,0.5449501276016235,0.3624723851680756,0,2,1,13,14,6,-1.,1,15,14,2,3.,-0.0806244164705276,0.1656087040901184,0.5000287294387817,0,2,13,6,1,6,-1.,13,9,1,3,2.,0.0221920292824507,0.5132731199264526,0.2002808004617691,0,2,0,4,3,8,-1.,1,4,1,8,3.,7.3100631125271320e-003,0.4617947936058044,0.6366536021232605,0,2,18,0,2,18,-1.,18,0,1,18,2.,-6.4063072204589844e-003,0.5916250944137573,0.4867860972881317,0,2,2,3,6,2,-1.,2,4,6,1,2.,-7.6415040530264378e-004,0.3888409137725830,0.5315797924995422,0,2,9,0,8,6,-1.,9,2,8,2,3.,7.6734489994123578e-004,0.4159064888954163,0.5605279803276062,0,2,6,6,1,6,-1.,6,9,1,3,2.,6.1474501853808761e-004,0.3089022040367127,0.5120148062705994,0,2,14,8,6,3,-1.,14,9,6,1,3.,-5.0105270929634571e-003,0.3972199857234955,0.5207306146621704,0,2,0,0,2,18,-1.,1,0,1,18,2.,-8.6909132078289986e-003,0.6257408261299133,0.4608575999736786,0,3,1,18,18,2,-1.,10,18,9,1,2.,1,19,9,1,2.,-0.0163914598524570,0.2085209935903549,0.5242266058921814,0,2,3,15,2,2,-1.,3,16,2,1,2.,4.0973909199237823e-004,0.5222427248954773,0.3780320882797241,0,2,8,14,5,3,-1.,8,15,5,1,3.,-2.5242289993911982e-003,0.5803927183151245,0.4611890017986298,0,2,8,14,2,3,-1.,8,15,2,1,3.,5.0945312250405550e-004,0.4401271939277649,0.5846015810966492,0,2,12,3,3,3,-1.,13,3,1,3,3.,1.9656419754028320e-003,0.5322325229644775,0.4184590876102448,0,2,7,5,6,2,-1.,9,5,2,2,3.,5.6298897834494710e-004,0.3741844892501831,0.5234565734863281,0,2,15,5,5,2,-1.,15,6,5,1,2.,-6.7946797935292125e-004,0.4631041884422302,0.5356478095054627,0,2,0,5,5,2,-1.,0,6,5,1,2.,7.2856349870562553e-003,0.5044670104980469,0.2377564013004303,0,2,17,14,1,6,-1.,17,17,1,3,2.,-0.0174594894051552,0.7289121150970459,0.5050435066223145,0,2,2,9,9,3,-1.,5,9,3,3,3.,-0.0254217498004436,0.6667134761810303,0.4678100049495697,0,2,12,3,3,3,-1.,13,3,1,3,3.,-1.5647639520466328e-003,0.4391759037971497,0.5323626995086670,0,2,0,0,4,18,-1.,2,0,2,18,2.,0.0114443600177765,0.4346440136432648,0.5680012106895447,0,2,17,6,1,3,-1.,17,7,1,1,3.,-6.7352550104260445e-004,0.4477140903472900,0.5296812057495117,0,2,2,14,1,6,-1.,2,17,1,3,2.,9.3194209039211273e-003,0.4740200042724609,0.7462607026100159,0,2,19,8,1,2,-1.,19,9,1,1,2.,1.3328490604180843e-004,0.5365061759948731,0.4752134978771210,0,2,5,3,3,3,-1.,6,3,1,3,3.,-7.8815799206495285e-003,0.1752219051122665,0.5015255212783814,0,2,9,16,2,3,-1.,9,17,2,1,3.,-5.7985680177807808e-003,0.7271236777305603,0.4896200895309448,0,2,2,6,1,3,-1.,2,7,1,1,3.,-3.8922499516047537e-004,0.4003908932209015,0.5344941020011902,0,3,12,4,8,2,-1.,16,4,4,1,2.,12,5,4,1,2.,-1.9288610201328993e-003,0.5605612993240356,0.4803955852985382,0,3,0,4,8,2,-1.,0,4,4,1,2.,4,5,4,1,2.,8.4214154630899429e-003,0.4753246903419495,0.7623608708381653,0,2,2,16,18,4,-1.,2,18,18,2,2.,8.1655876711010933e-003,0.5393261909484863,0.4191643893718720,0,2,7,15,2,4,-1.,7,17,2,2,2.,4.8280550981871784e-004,0.4240800142288208,0.5399821996688843,0,2,4,0,14,3,-1.,4,1,14,1,3.,-2.7186630759388208e-003,0.4244599938392639,0.5424923896789551,0,2,0,0,4,20,-1.,2,0,2,20,2.,-0.0125072300434113,0.5895841717720032,0.4550411105155945,0,3,12,4,4,8,-1.,14,4,2,4,2.,12,8,2,4,2.,-0.0242865197360516,0.2647134959697723,0.5189179778099060,0,3,6,7,2,2,-1.,6,7,1,1,2.,7,8,1,1,2.,-2.9676330741494894e-003,0.7347682714462280,0.4749749898910523,0,2,10,6,2,3,-1.,10,7,2,1,3.,-0.0125289997085929,0.2756049931049347,0.5177599787712097,0,2,8,7,3,2,-1.,8,8,3,1,2.,-1.0104000102728605e-003,0.3510560989379883,0.5144724249839783,0,2,8,2,6,12,-1.,8,8,6,6,2.,-2.1348530426621437e-003,0.5637925863265991,0.4667319953441620,0,2,4,0,11,12,-1.,4,4,11,4,3.,0.0195642597973347,0.4614573121070862,0.6137639880180359,0,2,14,9,6,11,-1.,16,9,2,11,3.,-0.0971463471651077,0.2998378872871399,0.5193555951118469,0,2,0,14,4,3,-1.,0,15,4,1,3.,4.5014568604528904e-003,0.5077884793281555,0.3045755922794342,0,2,9,10,2,3,-1.,9,11,2,1,3.,6.3706971704959869e-003,0.4861018955707550,0.6887500882148743,0,2,5,11,3,2,-1.,5,12,3,1,2.,-9.0721528977155685e-003,0.1673395931720734,0.5017563104629517,0,2,9,15,3,3,-1.,10,15,1,3,3.,-5.3537208586931229e-003,0.2692756950855255,0.5242633223533630,0,2,8,8,3,4,-1.,9,8,1,4,3.,-0.0109328404068947,0.7183864116668701,0.4736028909683228,0,2,9,15,3,3,-1.,10,15,1,3,3.,8.2356072962284088e-003,0.5223966836929321,0.2389862984418869,0,2,7,7,3,2,-1.,8,7,1,2,3.,-1.0038160253316164e-003,0.5719355940818787,0.4433943033218384,0,3,2,10,16,4,-1.,10,10,8,2,2.,2,12,8,2,2.,4.0859128348529339e-003,0.5472841858863831,0.4148836135864258,0,2,2,3,4,17,-1.,4,3,2,17,2.,0.1548541933298111,0.4973812103271484,0.0610615983605385,0,2,15,13,2,7,-1.,15,13,1,7,2.,2.0897459762636572e-004,0.4709174036979675,0.5423889160156250,0,2,2,2,6,1,-1.,5,2,3,1,2.,3.3316991175524890e-004,0.4089626967906952,0.5300992131233215,0,2,5,2,12,4,-1.,9,2,4,4,3.,-0.0108134001493454,0.6104369759559631,0.4957334101200104,0,3,6,0,8,12,-1.,6,0,4,6,2.,10,6,4,6,2.,0.0456560105085373,0.5069689154624939,0.2866660058498383,0,3,13,7,2,2,-1.,14,7,1,1,2.,13,8,1,1,2.,1.2569549726322293e-003,0.4846917092800140,0.6318171024322510,0,2,0,12,20,6,-1.,0,14,20,2,3.,-0.1201507002115250,0.0605261400341988,0.4980959892272949,0,2,14,7,2,3,-1.,14,7,1,3,2.,-1.0533799650147557e-004,0.5363109707832336,0.4708042144775391,0,2,0,8,9,12,-1.,3,8,3,12,3.,-0.2070319056510925,0.0596603304147720,0.4979098141193390,0,2,3,0,16,2,-1.,3,0,8,2,2.,1.2909180077258497e-004,0.4712977111339569,0.5377997756004334,0,2,6,15,3,3,-1.,6,16,3,1,3.,3.8818528992123902e-004,0.4363538026809692,0.5534191131591797,0,2,8,15,6,3,-1.,8,16,6,1,3.,-2.9243610333651304e-003,0.5811185836791992,0.4825215935707092,0,2,0,10,1,6,-1.,0,12,1,2,3.,8.3882332546636462e-004,0.5311700105667114,0.4038138985633850,0,2,10,9,4,3,-1.,10,10,4,1,3.,-1.9061550265178084e-003,0.3770701885223389,0.5260015130043030,0,2,9,15,2,3,-1.,9,16,2,1,3.,8.9514348655939102e-003,0.4766167998313904,0.7682183980941773,0,2,5,7,10,1,-1.,5,7,5,1,2.,0.0130834598094225,0.5264462828636169,0.3062222003936768,0,2,4,0,12,19,-1.,10,0,6,19,2.,-0.2115933001041412,0.6737198233604431,0.4695810079574585,0,3,0,6,20,6,-1.,10,6,10,3,2.,0,9,10,3,2.,3.1493250280618668e-003,0.5644835233688355,0.4386953115463257,0,3,3,6,2,2,-1.,3,6,1,1,2.,4,7,1,1,2.,3.9754100725986063e-004,0.4526061117649078,0.5895630121231079,0,3,15,6,2,2,-1.,16,6,1,1,2.,15,7,1,1,2.,-1.3814480043947697e-003,0.6070582270622253,0.4942413866519928,0,3,3,6,2,2,-1.,3,6,1,1,2.,4,7,1,1,2.,-5.8122188784182072e-004,0.5998213291168213,0.4508252143859863,0,2,14,4,1,12,-1.,14,10,1,6,2.,-2.3905329871922731e-003,0.4205588996410370,0.5223848223686218,0,3,2,5,16,10,-1.,2,5,8,5,2.,10,10,8,5,2.,0.0272689294070005,0.5206447243690491,0.3563301861286163,0,2,9,17,3,2,-1.,10,17,1,2,3.,-3.7658358924090862e-003,0.3144704103469849,0.5218814015388489,0,2,1,4,2,2,-1.,1,5,2,1,2.,-1.4903489500284195e-003,0.3380196094512940,0.5124437212944031,0,2,5,0,15,5,-1.,10,0,5,5,3.,-0.0174282304942608,0.5829960703849793,0.4919725954532623,0,2,0,0,15,5,-1.,5,0,5,5,3.,-0.0152780301868916,0.6163144707679749,0.4617887139320374,0,2,11,2,2,17,-1.,11,2,1,17,2.,0.0319956094026566,0.5166357159614563,0.1712764054536820,0,2,7,2,2,17,-1.,8,2,1,17,2.,-3.8256710395216942e-003,0.3408012092113495,0.5131387710571289,0,2,15,11,2,9,-1.,15,11,1,9,2.,-8.5186436772346497e-003,0.6105518937110901,0.4997941851615906,0,2,3,11,2,9,-1.,4,11,1,9,2.,9.0641621500253677e-004,0.4327270984649658,0.5582311153411865,0,2,5,16,14,4,-1.,5,16,7,4,2.,0.0103448498994112,0.4855653047561646,0.5452420115470886,79.2490768432617190,160,0,2,1,4,18,1,-1.,7,4,6,1,3.,7.8981826081871986e-003,0.3332524895668030,0.5946462154388428,0,3,13,7,6,4,-1.,16,7,3,2,2.,13,9,3,2,2.,1.6170160379260778e-003,0.3490641117095947,0.5577868819236755,0,2,9,8,2,12,-1.,9,12,2,4,3.,-5.5449741194024682e-004,0.5542566180229187,0.3291530013084412,0,2,12,1,6,6,-1.,12,3,6,2,3.,1.5428980113938451e-003,0.3612579107284546,0.5545979142189026,0,3,5,2,6,6,-1.,5,2,3,3,2.,8,5,3,3,2.,-1.0329450014978647e-003,0.3530139029026032,0.5576140284538269,0,3,9,16,6,4,-1.,12,16,3,2,2.,9,18,3,2,2.,7.7698158565908670e-004,0.3916778862476349,0.5645321011543274,0,2,1,2,18,3,-1.,7,2,6,3,3.,0.1432030051946640,0.4667482078075409,0.7023633122444153,0,2,7,4,9,10,-1.,7,9,9,5,2.,-7.3866490274667740e-003,0.3073684871196747,0.5289257764816284,0,2,5,9,4,4,-1.,7,9,2,4,2.,-6.2936742324382067e-004,0.5622118115425110,0.4037049114704132,0,2,11,10,3,6,-1.,11,13,3,3,2.,7.8893528552725911e-004,0.5267661213874817,0.3557874858379364,0,2,7,11,5,3,-1.,7,12,5,1,3.,-0.0122280502691865,0.6668320894241333,0.4625549912452698,0,3,7,11,6,6,-1.,10,11,3,3,2.,7,14,3,3,2.,3.5420239437371492e-003,0.5521438121795654,0.3869673013687134,0,2,0,0,10,9,-1.,0,3,10,3,3.,-1.0585320414975286e-003,0.3628678023815155,0.5320926904678345,0,2,13,14,1,6,-1.,13,16,1,2,3.,1.4935660146875307e-005,0.4632444977760315,0.5363323092460632,0,2,0,2,3,6,-1.,0,4,3,2,3.,5.2537708543241024e-003,0.5132231712341309,0.3265708982944489,0,2,8,14,4,3,-1.,8,15,4,1,3.,-8.2338023930788040e-003,0.6693689823150635,0.4774140119552612,0,2,6,14,1,6,-1.,6,16,1,2,3.,2.1866810129722580e-005,0.4053862094879150,0.5457931160926819,0,2,9,15,2,3,-1.,9,16,2,1,3.,-3.8150229956954718e-003,0.6454995870590210,0.4793178141117096,0,2,6,4,3,3,-1.,7,4,1,3,3.,1.1105879675596952e-003,0.5270407199859619,0.3529678881168366,0,2,9,0,11,3,-1.,9,1,11,1,3.,-5.7707689702510834e-003,0.3803547024726868,0.5352957844734192,0,2,0,6,20,3,-1.,0,7,20,1,3.,-3.0158339068293571e-003,0.5339403152465820,0.3887133002281189,0,2,10,1,1,2,-1.,10,2,1,1,2.,-8.5453689098358154e-004,0.3564616143703461,0.5273603796958923,0,2,9,6,2,6,-1.,10,6,1,6,2.,0.0110505102202296,0.4671907126903534,0.6849737763404846,0,2,5,8,12,1,-1.,9,8,4,1,3.,0.0426058396697044,0.5151473283767700,0.0702200904488564,0,2,3,8,12,1,-1.,7,8,4,1,3.,-3.0781750101596117e-003,0.3041661083698273,0.5152602195739746,0,2,9,7,3,5,-1.,10,7,1,5,3.,-5.4815728217363358e-003,0.6430295705795288,0.4897229969501495,0,2,3,9,6,2,-1.,6,9,3,2,2.,3.1881860923022032e-003,0.5307493209838867,0.3826209902763367,0,2,12,9,3,3,-1.,12,10,3,1,3.,3.5947180003859103e-004,0.4650047123432159,0.5421904921531677,0,2,7,0,6,1,-1.,9,0,2,1,3.,-4.0705031715333462e-003,0.2849679887294769,0.5079116225242615,0,2,12,9,3,3,-1.,12,10,3,1,3.,-0.0145941702648997,0.2971645891666412,0.5128461718559265,0,2,7,10,2,1,-1.,8,10,1,1,2.,-1.1947689927183092e-004,0.5631098151206970,0.4343082010746002,0,2,6,4,9,13,-1.,9,4,3,13,3.,-6.9344649091362953e-004,0.4403578042984009,0.5359959006309509,0,2,6,8,4,2,-1.,6,9,4,1,2.,1.4834799912932795e-005,0.3421008884906769,0.5164697766304016,0,2,16,2,4,6,-1.,16,2,2,6,2.,9.0296985581517220e-003,0.4639343023300171,0.6114075183868408,0,2,0,17,6,3,-1.,0,18,6,1,3.,-8.0640818923711777e-003,0.2820158898830414,0.5075494050979614,0,2,10,10,3,10,-1.,10,15,3,5,2.,0.0260621197521687,0.5208905935287476,0.2688778042793274,0,2,8,7,3,5,-1.,9,7,1,5,3.,0.0173146594315767,0.4663713872432709,0.6738539934158325,0,2,10,4,4,3,-1.,10,4,2,3,2.,0.0226666405797005,0.5209349989891052,0.2212723940610886,0,2,8,4,3,8,-1.,9,4,1,8,3.,-2.1965929772704840e-003,0.6063101291656494,0.4538190066814423,0,2,6,6,9,13,-1.,9,6,3,13,3.,-9.5282476395368576e-003,0.4635204970836639,0.5247430801391602,0,3,6,0,8,12,-1.,6,0,4,6,2.,10,6,4,6,2.,8.0943619832396507e-003,0.5289440155029297,0.3913882076740265,0,2,14,2,6,8,-1.,16,2,2,8,3.,-0.0728773325681686,0.7752001881599426,0.4990234971046448,0,2,6,0,3,6,-1.,7,0,1,6,3.,-6.9009521976113319e-003,0.2428039014339447,0.5048090219497681,0,2,14,2,6,8,-1.,16,2,2,8,3.,-0.0113082397729158,0.5734364986419678,0.4842376112937927,0,2,0,5,6,6,-1.,0,8,6,3,2.,0.0596132017672062,0.5029836297035217,0.2524977028369904,0,3,9,12,6,2,-1.,12,12,3,1,2.,9,13,3,1,2.,-2.8624620754271746e-003,0.6073045134544373,0.4898459911346436,0,2,8,17,3,2,-1.,9,17,1,2,3.,4.4781449250876904e-003,0.5015289187431335,0.2220316976308823,0,3,11,6,2,2,-1.,12,6,1,1,2.,11,7,1,1,2.,-1.7513240454718471e-003,0.6614428758621216,0.4933868944644928,0,2,1,9,18,2,-1.,7,9,6,2,3.,0.0401634201407433,0.5180878043174744,0.3741044998168945,0,3,11,6,2,2,-1.,12,6,1,1,2.,11,7,1,1,2.,3.4768949262797832e-004,0.4720416963100433,0.5818032026290894,0,2,3,4,12,8,-1.,7,4,4,8,3.,2.6551650371402502e-003,0.3805010914802551,0.5221335887908936,0,2,13,11,5,3,-1.,13,12,5,1,3.,-8.7706279009580612e-003,0.2944166064262390,0.5231295228004456,0,2,9,10,2,3,-1.,9,11,2,1,3.,-5.5122091434895992e-003,0.7346177101135254,0.4722816944122315,0,2,14,7,2,3,-1.,14,7,1,3,2.,6.8672042107209563e-004,0.5452876091003418,0.4242413043975830,0,2,5,4,1,3,-1.,5,5,1,1,3.,5.6019669864326715e-004,0.4398862123489380,0.5601285099983215,0,2,13,4,2,3,-1.,13,5,2,1,3.,2.4143769405782223e-003,0.4741686880588532,0.6136621832847595,0,2,5,4,2,3,-1.,5,5,2,1,3.,-1.5680900542065501e-003,0.6044552922248840,0.4516409933567047,0,2,9,8,2,3,-1.,9,9,2,1,3.,-3.6827491130679846e-003,0.2452459037303925,0.5294982194900513,0,2,8,9,2,2,-1.,8,10,2,1,2.,-2.9409190756268799e-004,0.3732838034629822,0.5251451134681702,0,2,15,14,1,4,-1.,15,16,1,2,2.,4.2847759323194623e-004,0.5498809814453125,0.4065535068511963,0,2,3,12,2,2,-1.,3,13,2,1,2.,-4.8817070201039314e-003,0.2139908969402313,0.4999957084655762,0,3,12,15,2,2,-1.,13,15,1,1,2.,12,16,1,1,2.,2.7272020815871656e-004,0.4650287032127380,0.5813428759574890,0,2,9,13,2,2,-1.,9,14,2,1,2.,2.0947199664078653e-004,0.4387486875057221,0.5572792887687683,0,2,4,11,14,9,-1.,4,14,14,3,3.,0.0485011897981167,0.5244972705841065,0.3212889134883881,0,2,7,13,4,3,-1.,7,14,4,1,3.,-4.5166411437094212e-003,0.6056813001632690,0.4545882046222687,0,2,15,14,1,4,-1.,15,16,1,2,2.,-0.0122916800901294,0.2040929049253464,0.5152214169502258,0,2,4,14,1,4,-1.,4,16,1,2,2.,4.8549679922871292e-004,0.5237604975700378,0.3739503026008606,0,2,14,0,6,13,-1.,16,0,2,13,3.,0.0305560491979122,0.4960533976554871,0.5938246250152588,0,3,4,1,2,12,-1.,4,1,1,6,2.,5,7,1,6,2.,-1.5105320198927075e-004,0.5351303815841675,0.4145204126834869,0,3,11,14,6,6,-1.,14,14,3,3,2.,11,17,3,3,2.,2.4937440175563097e-003,0.4693366885185242,0.5514941215515137,0,3,3,14,6,6,-1.,3,14,3,3,2.,6,17,3,3,2.,-0.0123821301385760,0.6791396737098694,0.4681667983531952,0,2,14,17,3,2,-1.,14,18,3,1,2.,-5.1333461888134480e-003,0.3608739078044891,0.5229160189628601,0,2,3,17,3,2,-1.,3,18,3,1,2.,5.1919277757406235e-004,0.5300073027610779,0.3633613884449005,0,2,14,0,6,13,-1.,16,0,2,13,3.,0.1506042033433914,0.5157316923141480,0.2211782038211823,0,2,0,0,6,13,-1.,2,0,2,13,3.,7.7144149690866470e-003,0.4410496950149536,0.5776609182357788,0,2,10,10,7,6,-1.,10,12,7,2,3.,9.4443522393703461e-003,0.5401855111122131,0.3756650090217590,0,3,6,15,2,2,-1.,6,15,1,1,2.,7,16,1,1,2.,2.5006249779835343e-004,0.4368270933628082,0.5607374906539917,0,3,6,11,8,6,-1.,10,11,4,3,2.,6,14,4,3,2.,-3.3077150583267212e-003,0.4244799017906189,0.5518230795860291,0,3,7,6,2,2,-1.,7,6,1,1,2.,8,7,1,1,2.,7.4048910755664110e-004,0.4496962130069733,0.5900576710700989,0,3,2,2,16,6,-1.,10,2,8,3,2.,2,5,8,3,2.,0.0440920516848564,0.5293493270874023,0.3156355023384094,0,2,5,4,3,3,-1.,5,5,3,1,3.,3.3639909233897924e-003,0.4483296871185303,0.5848662257194519,0,2,11,7,3,10,-1.,11,12,3,5,2.,-3.9760079234838486e-003,0.4559507071971893,0.5483639240264893,0,2,6,7,3,10,-1.,6,12,3,5,2.,2.7716930489987135e-003,0.5341786146163940,0.3792484104633331,0,2,10,7,3,2,-1.,11,7,1,2,3.,-2.4123019829858094e-004,0.5667188763618469,0.4576973021030426,0,2,8,12,4,2,-1.,8,13,4,1,2.,4.9425667384639382e-004,0.4421244859695435,0.5628787279129028,0,2,10,1,1,3,-1.,10,2,1,1,3.,-3.8876468897797167e-004,0.4288370907306671,0.5391063094139099,0,3,1,2,4,18,-1.,1,2,2,9,2.,3,11,2,9,2.,-0.0500488989055157,0.6899513006210327,0.4703742861747742,0,2,12,4,4,12,-1.,12,10,4,6,2.,-0.0366354808211327,0.2217779010534287,0.5191826224327087,0,2,0,0,1,6,-1.,0,2,1,2,3.,2.4273579474538565e-003,0.5136224031448364,0.3497397899627686,0,2,9,11,2,3,-1.,9,12,2,1,3.,1.9558030180633068e-003,0.4826192855834961,0.6408380866050720,0,2,8,7,4,3,-1.,8,8,4,1,3.,-1.7494610510766506e-003,0.3922835886478424,0.5272685289382935,0,2,10,7,3,2,-1.,11,7,1,2,3.,0.0139550799503922,0.5078201889991760,0.8416504859924316,0,2,7,7,3,2,-1.,8,7,1,2,3.,-2.1896739781368524e-004,0.5520489811897278,0.4314234852790833,0,2,9,4,6,1,-1.,11,4,2,1,3.,-1.5131309628486633e-003,0.3934605121612549,0.5382571220397949,0,2,8,7,2,3,-1.,9,7,1,3,2.,-4.3622800149023533e-003,0.7370628714561462,0.4736475944519043,0,3,12,7,8,6,-1.,16,7,4,3,2.,12,10,4,3,2.,0.0651605874300003,0.5159279704093933,0.3281595110893250,0,3,0,7,8,6,-1.,0,7,4,3,2.,4,10,4,3,2.,-2.3567399475723505e-003,0.3672826886177063,0.5172886252403259,0,3,18,2,2,10,-1.,19,2,1,5,2.,18,7,1,5,2.,0.0151466596871614,0.5031493902206421,0.6687604188919067,0,2,0,2,6,4,-1.,3,2,3,4,2.,-0.0228509604930878,0.6767519712448120,0.4709596931934357,0,2,9,4,6,1,-1.,11,4,2,1,3.,4.8867650330066681e-003,0.5257998108863831,0.4059878885746002,0,3,7,15,2,2,-1.,7,15,1,1,2.,8,16,1,1,2.,1.7619599821045995e-003,0.4696272909641266,0.6688278913497925,0,2,11,13,1,6,-1.,11,16,1,3,2.,-1.2942519970238209e-003,0.4320712983608246,0.5344281792640686,0,2,8,13,1,6,-1.,8,16,1,3,2.,0.0109299495816231,0.4997706115245819,0.1637486070394516,0,2,14,3,2,1,-1.,14,3,1,1,2.,2.9958489903947338e-005,0.4282417893409729,0.5633224248886108,0,2,8,15,2,3,-1.,8,16,2,1,3.,-6.5884361974895000e-003,0.6772121191024780,0.4700526893138886,0,2,12,15,7,4,-1.,12,17,7,2,2.,3.2527779694646597e-003,0.5313397049903870,0.4536148905754089,0,2,4,14,12,3,-1.,4,15,12,1,3.,-4.0435739792883396e-003,0.5660061836242676,0.4413388967514038,0,2,10,3,3,2,-1.,11,3,1,2,3.,-1.2523540062829852e-003,0.3731913864612579,0.5356451869010925,0,2,4,12,2,2,-1.,4,13,2,1,2.,1.9246719602961093e-004,0.5189986228942871,0.3738811016082764,0,2,10,11,4,6,-1.,10,14,4,3,2.,-0.0385896712541580,0.2956373989582062,0.5188810825347900,0,3,7,13,2,2,-1.,7,13,1,1,2.,8,14,1,1,2.,1.5489870565943420e-004,0.4347135126590729,0.5509533286094666,0,3,4,11,14,4,-1.,11,11,7,2,2.,4,13,7,2,2.,-0.0337638482451439,0.3230330049991608,0.5195475816726685,0,2,1,18,18,2,-1.,7,18,6,2,3.,-8.2657067105174065e-003,0.5975489020347595,0.4552114009857178,0,3,11,18,2,2,-1.,12,18,1,1,2.,11,19,1,1,2.,1.4481440302915871e-005,0.4745678007602692,0.5497426986694336,0,3,7,18,2,2,-1.,7,18,1,1,2.,8,19,1,1,2.,1.4951299817766994e-005,0.4324473142623901,0.5480644106864929,0,2,12,18,8,2,-1.,12,19,8,1,2.,-0.0187417995184660,0.1580052971839905,0.5178533196449280,0,2,7,14,6,2,-1.,7,15,6,1,2.,1.7572239739820361e-003,0.4517636895179749,0.5773764252662659,0,3,8,12,4,8,-1.,10,12,2,4,2.,8,16,2,4,2.,-3.1391119118779898e-003,0.4149647951126099,0.5460842251777649,0,2,4,9,3,3,-1.,4,10,3,1,3.,6.6656779381446540e-005,0.4039090871810913,0.5293084979057312,0,2,7,10,6,2,-1.,9,10,2,2,3.,6.7743421532213688e-003,0.4767651855945587,0.6121956110000610,0,2,5,0,4,15,-1.,7,0,2,15,2.,-7.3868161998689175e-003,0.3586258888244629,0.5187280774116516,0,2,8,6,12,14,-1.,12,6,4,14,3.,0.0140409301966429,0.4712139964103699,0.5576155781745911,0,2,5,16,3,3,-1.,5,17,3,1,3.,-5.5258329957723618e-003,0.2661027014255524,0.5039281249046326,0,2,8,1,12,19,-1.,12,1,4,19,3.,0.3868423998355866,0.5144339799880981,0.2525899112224579,0,2,3,0,3,2,-1.,3,1,3,1,2.,1.1459240340627730e-004,0.4284994900226593,0.5423371195793152,0,2,10,12,4,5,-1.,10,12,2,5,2.,-0.0184675697237253,0.3885835111141205,0.5213062167167664,0,2,6,12,4,5,-1.,8,12,2,5,2.,-4.5907011372037232e-004,0.5412563085556030,0.4235909879207611,0,3,11,11,2,2,-1.,12,11,1,1,2.,11,12,1,1,2.,1.2527540093287826e-003,0.4899305105209351,0.6624091267585754,0,2,0,2,3,6,-1.,0,4,3,2,3.,1.4910609461367130e-003,0.5286778211593628,0.4040051996707916,0,3,11,11,2,2,-1.,12,11,1,1,2.,11,12,1,1,2.,-7.5435562757775187e-004,0.6032990217208862,0.4795120060443878,0,2,7,6,4,10,-1.,7,11,4,5,2.,-6.9478838704526424e-003,0.4084401130676270,0.5373504161834717,0,3,11,11,2,2,-1.,12,11,1,1,2.,11,12,1,1,2.,2.8092920547351241e-004,0.4846062958240509,0.5759382247924805,0,2,2,13,5,2,-1.,2,14,5,1,2.,9.6073717577382922e-004,0.5164741277694702,0.3554979860782623,0,3,11,11,2,2,-1.,12,11,1,1,2.,11,12,1,1,2.,-2.6883929967880249e-004,0.5677582025527954,0.4731765985488892,0,3,7,11,2,2,-1.,7,11,1,1,2.,8,12,1,1,2.,2.1599370520561934e-003,0.4731487035751343,0.7070567011833191,0,2,14,13,3,3,-1.,14,14,3,1,3.,5.6235301308333874e-003,0.5240243077278137,0.2781791985034943,0,2,3,13,3,3,-1.,3,14,3,1,3.,-5.0243991427123547e-003,0.2837013900279999,0.5062304139137268,0,2,9,14,2,3,-1.,9,15,2,1,3.,-9.7611639648675919e-003,0.7400717735290527,0.4934569001197815,0,2,8,7,3,3,-1.,8,8,3,1,3.,4.1515100747346878e-003,0.5119131207466126,0.3407008051872253,0,2,13,5,3,3,-1.,13,6,3,1,3.,6.2465080991387367e-003,0.4923788011074066,0.6579058766365051,0,2,0,9,5,3,-1.,0,10,5,1,3.,-7.0597478188574314e-003,0.2434711009263992,0.5032842159271240,0,2,13,5,3,3,-1.,13,6,3,1,3.,-2.0587709732353687e-003,0.5900310873985291,0.4695087075233460,0,3,9,12,2,8,-1.,9,12,1,4,2.,10,16,1,4,2.,-2.4146060459315777e-003,0.3647317886352539,0.5189201831817627,0,3,11,7,2,2,-1.,12,7,1,1,2.,11,8,1,1,2.,-1.4817609917372465e-003,0.6034948229789734,0.4940128028392792,0,2,0,16,6,4,-1.,3,16,3,4,2.,-6.3016400672495365e-003,0.5818989872932434,0.4560427963733673,0,2,10,6,2,3,-1.,10,7,2,1,3.,3.4763428848236799e-003,0.5217475891113281,0.3483993113040924,0,2,9,5,2,6,-1.,9,7,2,2,3.,-0.0222508702427149,0.2360700070858002,0.5032082796096802,0,2,12,15,8,4,-1.,12,15,4,4,2.,-0.0306125506758690,0.6499186754226685,0.4914919137954712,0,2,0,14,8,6,-1.,4,14,4,6,2.,0.0130574796348810,0.4413323104381561,0.5683764219284058,0,2,9,0,3,2,-1.,10,0,1,2,3.,-6.0095742810517550e-004,0.4359731078147888,0.5333483219146729,0,2,4,15,4,2,-1.,6,15,2,2,2.,-4.1514250915497541e-004,0.5504062771797180,0.4326060116291046,0,2,12,7,3,13,-1.,13,7,1,13,3.,-0.0137762902304530,0.4064112901687622,0.5201548933982849,0,2,5,7,3,13,-1.,6,7,1,13,3.,-0.0322965085506439,0.0473519712686539,0.4977194964885712,0,2,9,6,3,9,-1.,9,9,3,3,3.,0.0535569787025452,0.4881733059883118,0.6666939258575440,0,2,4,4,7,12,-1.,4,10,7,6,2.,8.1889545544981956e-003,0.5400037169456482,0.4240820109844208,0,3,12,12,2,2,-1.,13,12,1,1,2.,12,13,1,1,2.,2.1055320394225419e-004,0.4802047908306122,0.5563852787017822,0,3,6,12,2,2,-1.,6,12,1,1,2.,7,13,1,1,2.,-2.4382730480283499e-003,0.7387793064117432,0.4773685038089752,0,3,8,9,4,2,-1.,10,9,2,1,2.,8,10,2,1,2.,3.2835570164024830e-003,0.5288546085357666,0.3171291947364807,0,3,3,6,2,2,-1.,3,6,1,1,2.,4,7,1,1,2.,2.3729570675641298e-003,0.4750812947750092,0.7060170769691467,0,2,16,6,3,2,-1.,16,7,3,1,2.,-1.4541699783876538e-003,0.3811730146408081,0.5330739021301270,87.6960296630859380,177,0,2,0,7,19,4,-1.,0,9,19,2,2.,0.0557552389800549,0.4019156992435455,0.6806036829948425,0,2,10,2,10,1,-1.,10,2,5,1,2.,2.4730248842388391e-003,0.3351148962974548,0.5965719819068909,0,2,9,4,2,12,-1.,9,10,2,6,2.,-3.5031698644161224e-004,0.5557708144187927,0.3482286930084229,0,2,12,18,4,1,-1.,12,18,2,1,2.,5.4167630150914192e-004,0.4260858893394470,0.5693380832672119,0,3,1,7,6,4,-1.,1,7,3,2,2.,4,9,3,2,2.,7.7193678589537740e-004,0.3494240045547485,0.5433688759803772,0,2,12,0,6,13,-1.,14,0,2,13,3.,-1.5999219613149762e-003,0.4028499126434326,0.5484359264373779,0,2,2,0,6,13,-1.,4,0,2,13,3.,-1.1832080053864047e-004,0.3806901872158051,0.5425465106964111,0,2,10,5,8,8,-1.,10,9,8,4,2.,3.2909031142480671e-004,0.2620100080966950,0.5429521799087524,0,2,8,3,2,5,-1.,9,3,1,5,2.,2.9518108931370080e-004,0.3799768984317780,0.5399264097213745,0,2,8,4,9,1,-1.,11,4,3,1,3.,9.0466710389591753e-005,0.4433645009994507,0.5440226197242737,0,2,3,4,9,1,-1.,6,4,3,1,3.,1.5007190086180344e-005,0.3719654977321625,0.5409119725227356,0,2,1,0,18,10,-1.,7,0,6,10,3.,0.1393561065196991,0.5525395870208740,0.4479042887687683,0,2,7,17,5,3,-1.,7,18,5,1,3.,1.6461990308016539e-003,0.4264501035213471,0.5772169828414917,0,2,7,11,6,1,-1.,9,11,2,1,3.,4.9984431825578213e-004,0.4359526038169861,0.5685871243476868,0,2,2,2,3,2,-1.,2,3,3,1,2.,-1.0971280280500650e-003,0.3390136957168579,0.5205408930778503,0,2,8,12,4,2,-1.,8,13,4,1,2.,6.6919892560690641e-004,0.4557456076145172,0.5980659723281860,0,2,6,10,3,6,-1.,6,13,3,3,2.,8.6471042595803738e-004,0.5134841203689575,0.2944033145904541,0,2,11,4,2,4,-1.,11,4,1,4,2.,-2.7182599296793342e-004,0.3906578123569489,0.5377181172370911,0,2,7,4,2,4,-1.,8,4,1,4,2.,3.0249499104684219e-005,0.3679609894752502,0.5225688815116882,0,2,9,6,2,4,-1.,9,6,1,4,2.,-8.5225896909832954e-003,0.7293102145195007,0.4892365038394928,0,2,6,13,8,3,-1.,6,14,8,1,3.,1.6705560265108943e-003,0.4345324933528900,0.5696138143539429,0,2,9,15,3,4,-1.,10,15,1,4,3.,-7.1433838456869125e-003,0.2591280043125153,0.5225623846054077,0,2,9,2,2,17,-1.,10,2,1,17,2.,-0.0163193698972464,0.6922279000282288,0.4651575982570648,0,2,7,0,6,1,-1.,9,0,2,1,3.,4.8034260980784893e-003,0.5352262854576111,0.3286302983760834,0,2,8,15,3,4,-1.,9,15,1,4,3.,-7.5421929359436035e-003,0.2040544003248215,0.5034546256065369,0,2,7,13,7,3,-1.,7,14,7,1,3.,-0.0143631100654602,0.6804888844490051,0.4889059066772461,0,2,8,16,3,3,-1.,9,16,1,3,3.,8.9063588529825211e-004,0.5310695767402649,0.3895480930805206,0,2,6,2,8,10,-1.,6,7,8,5,2.,-4.4060191139578819e-003,0.5741562843322754,0.4372426867485046,0,2,2,5,8,8,-1.,2,9,8,4,2.,-1.8862540309783071e-004,0.2831785976886749,0.5098205208778381,0,2,14,16,2,2,-1.,14,17,2,1,2.,-3.7979281041771173e-003,0.3372507989406586,0.5246580243110657,0,2,4,16,2,2,-1.,4,17,2,1,2.,1.4627049677073956e-004,0.5306674242019653,0.3911710083484650,0,2,10,11,4,6,-1.,10,14,4,3,2.,-4.9164638767251745e-005,0.5462496280670166,0.3942720890045166,0,2,6,11,4,6,-1.,6,14,4,3,2.,-0.0335825011134148,0.2157824039459229,0.5048211812973023,0,2,10,14,1,3,-1.,10,15,1,1,3.,-3.5339309833943844e-003,0.6465312242507935,0.4872696995735169,0,2,8,14,4,3,-1.,8,15,4,1,3.,5.0144111737608910e-003,0.4617668092250824,0.6248074769973755,0,3,10,0,4,6,-1.,12,0,2,3,2.,10,3,2,3,2.,0.0188173707574606,0.5220689177513123,0.2000052034854889,0,2,0,3,20,2,-1.,0,4,20,1,2.,-1.3434339780360460e-003,0.4014537930488586,0.5301619768142700,0,3,12,0,8,2,-1.,16,0,4,1,2.,12,1,4,1,2.,1.7557960236445069e-003,0.4794039130210877,0.5653169751167297,0,2,2,12,10,8,-1.,2,16,10,4,2.,-0.0956374630331993,0.2034195065498352,0.5006706714630127,0,3,17,7,2,10,-1.,18,7,1,5,2.,17,12,1,5,2.,-0.0222412291914225,0.7672473192214966,0.5046340227127075,0,3,1,7,2,10,-1.,1,7,1,5,2.,2,12,1,5,2.,-0.0155758196488023,0.7490342259407044,0.4755851030349731,0,2,15,10,3,6,-1.,15,12,3,2,3.,5.3599118255078793e-003,0.5365303754806519,0.4004670977592468,0,2,4,4,6,2,-1.,6,4,2,2,3.,-0.0217634998261929,0.0740154981613159,0.4964174926280975,0,2,0,5,20,6,-1.,0,7,20,2,3.,-0.1656159013509750,0.2859103083610535,0.5218086242675781,0,3,0,0,8,2,-1.,0,0,4,1,2.,4,1,4,1,2.,1.6461320046801120e-004,0.4191615879535675,0.5380793213844299,0,2,1,0,18,4,-1.,7,0,6,4,3.,-8.9077502489089966e-003,0.6273192763328552,0.4877404868602753,0,2,1,13,6,2,-1.,1,14,6,1,2.,8.6346449097618461e-004,0.5159940719604492,0.3671025931835175,0,2,10,8,3,4,-1.,11,8,1,4,3.,-1.3751760125160217e-003,0.5884376764297485,0.4579083919525147,0,2,6,1,6,1,-1.,8,1,2,1,3.,-1.4081239933148026e-003,0.3560509979724884,0.5139945149421692,0,2,8,14,4,3,-1.,8,15,4,1,3.,-3.9342888630926609e-003,0.5994288921356201,0.4664272069931030,0,2,1,6,18,2,-1.,10,6,9,2,2.,-0.0319669283926487,0.3345462083816528,0.5144183039665222,0,2,15,11,1,2,-1.,15,12,1,1,2.,-1.5089280168467667e-005,0.5582656264305115,0.4414057135581970,0,2,6,5,1,2,-1.,6,6,1,1,2.,5.1994470413774252e-004,0.4623680114746094,0.6168993711471558,0,2,13,4,1,3,-1.,13,5,1,1,3.,-3.4220460802316666e-003,0.6557074785232544,0.4974805116653442,0,2,2,15,1,2,-1.,2,16,1,1,2.,1.7723299970384687e-004,0.5269501805305481,0.3901908099651337,0,2,12,4,4,3,-1.,12,5,4,1,3.,1.5716759953647852e-003,0.4633373022079468,0.5790457725524902,0,2,0,0,7,3,-1.,0,1,7,1,3.,-8.9041329920291901e-003,0.2689608037471771,0.5053591132164002,0,2,9,12,6,2,-1.,9,12,3,2,2.,4.0677518700249493e-004,0.5456603169441223,0.4329898953437805,0,2,5,4,2,3,-1.,5,5,2,1,3.,6.7604780197143555e-003,0.4648993909358978,0.6689761877059937,0,2,18,4,2,3,-1.,18,5,2,1,3.,2.9100088868290186e-003,0.5309703946113586,0.3377839922904968,0,2,3,0,8,6,-1.,3,2,8,2,3.,1.3885459629818797e-003,0.4074738919734955,0.5349133014678955,0,3,0,2,20,6,-1.,10,2,10,3,2.,0,5,10,3,2.,-0.0767642632126808,0.1992176026105881,0.5228242278099060,0,2,4,7,2,4,-1.,5,7,1,4,2.,-2.2688310127705336e-004,0.5438501834869385,0.4253072142601013,0,2,3,10,15,2,-1.,8,10,5,2,3.,-6.3094152137637138e-003,0.4259178936481476,0.5378909707069397,0,2,3,0,12,11,-1.,9,0,6,11,2.,-0.1100727990269661,0.6904156804084778,0.4721749126911163,0,2,13,0,2,6,-1.,13,0,1,6,2.,2.8619659133255482e-004,0.4524914920330048,0.5548306107521057,0,2,0,19,2,1,-1.,1,19,1,1,2.,2.9425329557852820e-005,0.5370373725891113,0.4236463904380798,0,3,16,10,4,10,-1.,18,10,2,5,2.,16,15,2,5,2.,-0.0248865708708763,0.6423557996749878,0.4969303905963898,0,2,4,8,10,3,-1.,4,9,10,1,3.,0.0331488512456417,0.4988475143909454,0.1613811999559403,0,2,14,12,3,3,-1.,14,13,3,1,3.,7.8491691965609789e-004,0.5416026115417481,0.4223009049892426,0,3,0,10,4,10,-1.,0,10,2,5,2.,2,15,2,5,2.,4.7087189741432667e-003,0.4576328992843628,0.6027557849884033,0,2,18,3,2,6,-1.,18,5,2,2,3.,2.4144479539245367e-003,0.5308973193168640,0.4422498941421509,0,2,6,6,1,3,-1.,6,7,1,1,3.,1.9523180089890957e-003,0.4705634117126465,0.6663324832916260,0,2,7,7,7,2,-1.,7,8,7,1,2.,1.3031980488449335e-003,0.4406126141548157,0.5526962280273438,0,2,0,3,2,6,-1.,0,5,2,2,3.,4.4735497795045376e-003,0.5129023790359497,0.3301498889923096,0,2,11,1,3,1,-1.,12,1,1,1,3.,-2.6652868837118149e-003,0.3135471045970917,0.5175036191940308,0,2,5,0,2,6,-1.,6,0,1,6,2.,1.3666770246345550e-004,0.4119370877742767,0.5306876897811890,0,2,1,1,18,14,-1.,7,1,6,14,3.,-0.0171264503151178,0.6177806258201599,0.4836578965187073,0,2,4,6,8,3,-1.,8,6,4,3,2.,-2.6601430727168918e-004,0.3654330968856812,0.5169736742973328,0,2,9,12,6,2,-1.,9,12,3,2,2.,-0.0229323804378510,0.3490915000438690,0.5163992047309876,0,2,5,12,6,2,-1.,8,12,3,2,2.,2.3316550068557262e-003,0.5166299939155579,0.3709389865398407,0,2,10,7,3,5,-1.,11,7,1,5,3.,0.0169256608933210,0.5014736056327820,0.8053988218307495,0,2,7,7,3,5,-1.,8,7,1,5,3.,-8.9858826249837875e-003,0.6470788717269898,0.4657020866870880,0,2,13,0,3,10,-1.,14,0,1,10,3.,-0.0118746999651194,0.3246378898620606,0.5258755087852478,0,2,4,11,3,2,-1.,4,12,3,1,2.,1.9350569345988333e-004,0.5191941857337952,0.3839643895626068,0,2,17,3,3,6,-1.,18,3,1,6,3.,5.8713490143418312e-003,0.4918133914470673,0.6187043190002441,0,2,1,8,18,10,-1.,1,13,18,5,2.,-0.2483879029750824,0.1836802959442139,0.4988150000572205,0,2,13,0,3,10,-1.,14,0,1,10,3.,0.0122560001909733,0.5227053761482239,0.3632029891014099,0,2,9,14,2,3,-1.,9,15,2,1,3.,8.3990179700776935e-004,0.4490250051021576,0.5774148106575012,0,2,16,3,3,7,-1.,17,3,1,7,3.,2.5407369248569012e-003,0.4804787039756775,0.5858299136161804,0,2,4,0,3,10,-1.,5,0,1,10,3.,-0.0148224299773574,0.2521049976348877,0.5023537278175354,0,2,16,3,3,7,-1.,17,3,1,7,3.,-5.7973959483206272e-003,0.5996695756912231,0.4853715002536774,0,2,0,9,1,2,-1.,0,10,1,1,2.,7.2662148158997297e-004,0.5153716802597046,0.3671779930591583,0,2,18,1,2,10,-1.,18,1,1,10,2.,-0.0172325801104307,0.6621719002723694,0.4994656145572662,0,2,0,1,2,10,-1.,1,1,1,10,2.,7.8624086454510689e-003,0.4633395075798035,0.6256101727485657,0,2,10,16,3,4,-1.,11,16,1,4,3.,-4.7343620099127293e-003,0.3615573048591614,0.5281885266304016,0,2,2,8,3,3,-1.,3,8,1,3,3.,8.3048478700220585e-004,0.4442889094352722,0.5550957918167114,0,3,11,0,2,6,-1.,12,0,1,3,2.,11,3,1,3,2.,7.6602199114859104e-003,0.5162935256958008,0.2613354921340942,0,3,7,0,2,6,-1.,7,0,1,3,2.,8,3,1,3,2.,-4.1048377752304077e-003,0.2789632081985474,0.5019031763076782,0,2,16,3,3,7,-1.,17,3,1,7,3.,4.8512578941881657e-003,0.4968984127044678,0.5661668181419373,0,2,1,3,3,7,-1.,2,3,1,7,3.,9.9896453320980072e-004,0.4445607960224152,0.5551813244819641,0,2,14,1,6,16,-1.,16,1,2,16,3.,-0.2702363133430481,0.0293882098048925,0.5151314139366150,0,2,0,1,6,16,-1.,2,1,2,16,3.,-0.0130906803533435,0.5699399709701538,0.4447459876537323,0,3,2,0,16,8,-1.,10,0,8,4,2.,2,4,8,4,2.,-9.4342790544033051e-003,0.4305466115474701,0.5487895011901856,0,2,6,8,5,3,-1.,6,9,5,1,3.,-1.5482039889320731e-003,0.3680317103862763,0.5128080844879150,0,2,9,7,3,3,-1.,10,7,1,3,3.,5.3746132180094719e-003,0.4838916957378388,0.6101555824279785,0,2,8,8,4,3,-1.,8,9,4,1,3.,1.5786769799888134e-003,0.5325223207473755,0.4118548035621643,0,2,9,6,2,4,-1.,9,6,1,4,2.,3.6856050137430429e-003,0.4810948073863983,0.6252303123474121,0,2,0,7,15,1,-1.,5,7,5,1,3.,9.3887019902467728e-003,0.5200229883193970,0.3629410862922669,0,2,8,2,7,9,-1.,8,5,7,3,3.,0.0127926301211119,0.4961709976196289,0.6738016009330750,0,3,1,7,16,4,-1.,1,7,8,2,2.,9,9,8,2,2.,-3.3661040943115950e-003,0.4060279130935669,0.5283598899841309,0,2,6,12,8,2,-1.,6,13,8,1,2.,3.9771420415490866e-004,0.4674113988876343,0.5900775194168091,0,2,8,11,3,3,-1.,8,12,3,1,3.,1.4868030557408929e-003,0.4519116878509522,0.6082053780555725,0,3,4,5,14,10,-1.,11,5,7,5,2.,4,10,7,5,2.,-0.0886867493391037,0.2807899117469788,0.5180991888046265,0,2,4,12,3,2,-1.,4,13,3,1,2.,-7.4296112870797515e-005,0.5295584201812744,0.4087625145912170,0,2,9,11,6,1,-1.,11,11,2,1,3.,-1.4932939848222304e-005,0.5461400151252747,0.4538542926311493,0,2,4,9,7,6,-1.,4,11,7,2,3.,5.9162238612771034e-003,0.5329161286354065,0.4192134141921997,0,2,7,10,6,3,-1.,7,11,6,1,3.,1.1141640134155750e-003,0.4512017965316773,0.5706217288970947,0,2,9,11,2,2,-1.,9,12,2,1,2.,8.9249362645205110e-005,0.4577805995941162,0.5897638201713562,0,2,0,5,20,6,-1.,0,7,20,2,3.,2.5319510605186224e-003,0.5299603939056397,0.3357639014720917,0,2,6,4,6,1,-1.,8,4,2,1,3.,0.0124262003228068,0.4959059059619904,0.1346601992845535,0,2,9,11,6,1,-1.,11,11,2,1,3.,0.0283357501029968,0.5117079019546509,6.1043637106195092e-004,0,2,5,11,6,1,-1.,7,11,2,1,3.,6.6165882162749767e-003,0.4736349880695343,0.7011628150939941,0,2,10,16,3,4,-1.,11,16,1,4,3.,8.0468766391277313e-003,0.5216417908668518,0.3282819986343384,0,2,8,7,3,3,-1.,9,7,1,3,3.,-1.1193980462849140e-003,0.5809860825538635,0.4563739001750946,0,2,2,12,16,8,-1.,2,16,16,4,2.,0.0132775902748108,0.5398362278938294,0.4103901088237763,0,2,0,15,15,2,-1.,0,16,15,1,2.,4.8794739996083081e-004,0.4249286055564880,0.5410590767860413,0,2,15,4,5,6,-1.,15,6,5,2,3.,0.0112431701272726,0.5269963741302490,0.3438215851783752,0,2,9,5,2,4,-1.,10,5,1,4,2.,-8.9896668214350939e-004,0.5633075833320618,0.4456613063812256,0,2,8,10,9,6,-1.,8,12,9,2,3.,6.6677159629762173e-003,0.5312889218330383,0.4362679123878479,0,2,2,19,15,1,-1.,7,19,5,1,3.,0.0289472993463278,0.4701794981956482,0.6575797796249390,0,2,10,16,3,4,-1.,11,16,1,4,3.,-0.0234000496566296,0.,0.5137398838996887,0,2,0,15,20,4,-1.,0,17,20,2,2.,-0.0891170501708984,0.0237452797591686,0.4942430853843689,0,2,10,16,3,4,-1.,11,16,1,4,3.,-0.0140546001493931,0.3127323091030121,0.5117511153221130,0,2,7,16,3,4,-1.,8,16,1,4,3.,8.1239398568868637e-003,0.5009049177169800,0.2520025968551636,0,2,9,16,3,3,-1.,9,17,3,1,3.,-4.9964650534093380e-003,0.6387143731117249,0.4927811920642853,0,2,8,11,4,6,-1.,8,14,4,3,2.,3.1253970228135586e-003,0.5136849880218506,0.3680452108383179,0,2,9,6,2,12,-1.,9,10,2,4,3.,6.7669642157852650e-003,0.5509843826293945,0.4363631904125214,0,2,8,17,4,3,-1.,8,18,4,1,3.,-2.3711440153419971e-003,0.6162335276603699,0.4586946964263916,0,3,9,18,8,2,-1.,13,18,4,1,2.,9,19,4,1,2.,-5.3522791713476181e-003,0.6185457706451416,0.4920490980148315,0,2,1,18,8,2,-1.,1,19,8,1,2.,-0.0159688591957092,0.1382617950439453,0.4983252882957459,0,2,13,5,6,15,-1.,15,5,2,15,3.,4.7676060348749161e-003,0.4688057899475098,0.5490046143531799,0,2,9,8,2,2,-1.,9,9,2,1,2.,-2.4714691098779440e-003,0.2368514984846115,0.5003952980041504,0,2,9,5,2,3,-1.,9,5,1,3,2.,-7.1033788844943047e-004,0.5856394171714783,0.4721533060073853,0,2,1,5,6,15,-1.,3,5,2,15,3.,-0.1411755979061127,0.0869000628590584,0.4961591064929962,0,3,4,1,14,8,-1.,11,1,7,4,2.,4,5,7,4,2.,0.1065180972218514,0.5138837099075317,0.1741005033254623,0,3,2,4,4,16,-1.,2,4,2,8,2.,4,12,2,8,2.,-0.0527447499334812,0.7353636026382446,0.4772881865501404,0,2,12,4,3,12,-1.,12,10,3,6,2.,-4.7431760467588902e-003,0.3884406089782715,0.5292701721191406,0,3,4,5,10,12,-1.,4,5,5,6,2.,9,11,5,6,2.,9.9676765967160463e-004,0.5223492980003357,0.4003424048423767,0,2,9,14,2,3,-1.,9,15,2,1,3.,8.0284131690859795e-003,0.4959106147289276,0.7212964296340942,0,2,5,4,2,3,-1.,5,5,2,1,3.,8.6025858763605356e-004,0.4444884061813355,0.5538476109504700,0,3,12,2,4,10,-1.,14,2,2,5,2.,12,7,2,5,2.,9.3191501218825579e-004,0.5398371219635010,0.4163244068622589,0,2,6,4,7,3,-1.,6,5,7,1,3.,-2.5082060601562262e-003,0.5854265093803406,0.4562500119209290,0,3,2,0,18,2,-1.,11,0,9,1,2.,2,1,9,1,2.,-2.1378761157393456e-003,0.4608069062232971,0.5280259251594544,0,3,0,0,18,2,-1.,0,0,9,1,2.,9,1,9,1,2.,-2.1546049974858761e-003,0.3791126906871796,0.5255997180938721,0,3,13,13,4,6,-1.,15,13,2,3,2.,13,16,2,3,2.,-7.6214009895920753e-003,0.5998609066009522,0.4952073991298676,0,3,3,13,4,6,-1.,3,13,2,3,2.,5,16,2,3,2.,2.2055360022932291e-003,0.4484206140041351,0.5588530898094177,0,2,10,12,2,6,-1.,10,15,2,3,2.,1.2586950324475765e-003,0.5450747013092041,0.4423840939998627,0,3,5,9,10,10,-1.,5,9,5,5,2.,10,14,5,5,2.,-5.0926720723509789e-003,0.4118275046348572,0.5263035893440247,0,3,11,4,4,2,-1.,13,4,2,1,2.,11,5,2,1,2.,-2.5095739401876926e-003,0.5787907838821411,0.4998494982719421,0,2,7,12,6,8,-1.,10,12,3,8,2.,-0.0773275569081306,0.8397865891456604,0.4811120033264160,0,3,12,2,4,10,-1.,14,2,2,5,2.,12,7,2,5,2.,-0.0414858199656010,0.2408611029386520,0.5176993012428284,0,2,8,11,2,1,-1.,9,11,1,1,2.,1.0355669655837119e-004,0.4355360865592957,0.5417054295539856,0,2,10,5,1,12,-1.,10,9,1,4,3.,1.3255809899419546e-003,0.5453971028327942,0.4894095063209534,0,2,0,11,6,9,-1.,3,11,3,9,2.,-8.0598732456564903e-003,0.5771024227142334,0.4577918946743012,0,3,12,2,4,10,-1.,14,2,2,5,2.,12,7,2,5,2.,0.0190586205571890,0.5169867873191834,0.3400475084781647,0,3,4,2,4,10,-1.,4,2,2,5,2.,6,7,2,5,2.,-0.0350578911602497,0.2203243970870972,0.5000503063201904,0,3,11,4,4,2,-1.,13,4,2,1,2.,11,5,2,1,2.,5.7296059094369411e-003,0.5043408274650574,0.6597570776939392,0,2,0,14,6,3,-1.,0,15,6,1,3.,-0.0116483299061656,0.2186284959316254,0.4996652901172638,0,3,11,4,4,2,-1.,13,4,2,1,2.,11,5,2,1,2.,1.4544479781761765e-003,0.5007681846618652,0.5503727793693543,0,2,6,1,3,2,-1.,7,1,1,2,3.,-2.5030909455381334e-004,0.4129841029644013,0.5241670012474060,0,3,11,4,4,2,-1.,13,4,2,1,2.,11,5,2,1,2.,-8.2907272735610604e-004,0.5412868261337280,0.4974496066570282,0,3,5,4,4,2,-1.,5,4,2,1,2.,7,5,2,1,2.,1.0862209601327777e-003,0.4605529904365540,0.5879228711128235,0,3,13,0,2,12,-1.,14,0,1,6,2.,13,6,1,6,2.,2.0000500080641359e-004,0.5278854966163635,0.4705209136009216,0,2,6,0,3,10,-1.,7,0,1,10,3.,2.9212920926511288e-003,0.5129609704017639,0.3755536973476410,0,2,3,0,17,8,-1.,3,4,17,4,2.,0.0253874007612467,0.4822691977024078,0.5790768265724182,0,2,0,4,20,4,-1.,0,6,20,2,2.,-3.1968469265848398e-003,0.5248395204544067,0.3962840139865875,90.2533493041992190,182,0,2,0,3,8,2,-1.,4,3,4,2,2.,5.8031738735735416e-003,0.3498983979225159,0.5961983203887940,0,2,8,11,4,3,-1.,8,12,4,1,3.,-9.0003069490194321e-003,0.6816636919975281,0.4478552043437958,0,3,5,7,6,4,-1.,5,7,3,2,2.,8,9,3,2,2.,-1.1549659539014101e-003,0.5585706233978272,0.3578251004219055,0,2,8,3,4,9,-1.,8,6,4,3,3.,-1.1069850297644734e-003,0.5365036129951477,0.3050428032875061,0,2,8,15,1,4,-1.,8,17,1,2,2.,1.0308309720130637e-004,0.3639095127582550,0.5344635844230652,0,2,4,5,12,7,-1.,8,5,4,7,3.,-5.0984839908778667e-003,0.2859157025814056,0.5504264831542969,0,3,4,2,4,10,-1.,4,2,2,5,2.,6,7,2,5,2.,8.2572200335562229e-004,0.5236523747444153,0.3476041853427887,0,2,3,0,17,2,-1.,3,1,17,1,2.,9.9783325567841530e-003,0.4750322103500366,0.6219646930694580,0,2,2,2,16,15,-1.,2,7,16,5,3.,-0.0374025292694569,0.3343375921249390,0.5278062820434570,0,2,15,2,5,2,-1.,15,3,5,1,2.,4.8548257909715176e-003,0.5192180871963501,0.3700444102287293,0,2,9,3,2,2,-1.,10,3,1,2,2.,-1.8664470408111811e-003,0.2929843962192535,0.5091944932937622,0,2,4,5,16,15,-1.,4,10,16,5,3.,0.0168888904154301,0.3686845898628235,0.5431225895881653,0,2,7,13,5,6,-1.,7,16,5,3,2.,-5.8372621424496174e-003,0.3632183969020844,0.5221335887908936,0,2,10,7,3,2,-1.,11,7,1,2,3.,-1.4713739510625601e-003,0.5870683789253235,0.4700650870800018,0,2,8,3,3,1,-1.,9,3,1,1,3.,-1.1522950371727347e-003,0.3195894956588745,0.5140954256057739,0,2,9,16,3,3,-1.,9,17,3,1,3.,-4.2560300789773464e-003,0.6301859021186829,0.4814921021461487,0,2,0,2,5,2,-1.,0,3,5,1,2.,-6.7378291860222816e-003,0.1977048069238663,0.5025808215141296,0,2,12,5,4,3,-1.,12,6,4,1,3.,0.0113826701417565,0.4954132139682770,0.6867045760154724,0,2,1,7,12,1,-1.,5,7,4,1,3.,5.1794708706438541e-003,0.5164427757263184,0.3350647985935211,0,2,7,5,6,14,-1.,7,12,6,7,2.,-0.1174378991127014,0.2315246015787125,0.5234413743019104,0,3,0,0,8,10,-1.,0,0,4,5,2.,4,5,4,5,2.,0.0287034492939711,0.4664297103881836,0.6722521185874939,0,2,9,1,3,2,-1.,10,1,1,2,3.,4.8231030814349651e-003,0.5220875144004822,0.2723532915115356,0,2,8,1,3,2,-1.,9,1,1,2,3.,2.6798530016094446e-003,0.5079277157783508,0.2906948924064636,0,2,12,4,3,3,-1.,12,5,3,1,3.,8.0504082143306732e-003,0.4885950982570648,0.6395021080970764,0,2,7,4,6,16,-1.,7,12,6,8,2.,4.8054959625005722e-003,0.5197256803512573,0.3656663894653320,0,2,12,4,3,3,-1.,12,5,3,1,3.,-2.2420159075409174e-003,0.6153467893600464,0.4763701856136322,0,2,2,3,2,6,-1.,2,5,2,2,3.,-0.0137577103450894,0.2637344896793366,0.5030903220176697,0,2,14,2,6,9,-1.,14,5,6,3,3.,-0.1033829972147942,0.2287521958351135,0.5182461142539978,0,2,5,4,3,3,-1.,5,5,3,1,3.,-9.4432085752487183e-003,0.6953303813934326,0.4694949090480804,0,2,9,17,3,2,-1.,10,17,1,2,3.,8.0271181650459766e-004,0.5450655221939087,0.4268783926963806,0,2,5,5,2,3,-1.,5,6,2,1,3.,-4.1945669800043106e-003,0.6091387867927551,0.4571642875671387,0,2,13,11,3,6,-1.,13,13,3,2,3.,0.0109422104433179,0.5241063237190247,0.3284547030925751,0,2,3,14,2,6,-1.,3,17,2,3,2.,-5.7841069065034389e-004,0.5387929081916809,0.4179368913173676,0,2,14,3,6,2,-1.,14,4,6,1,2.,-2.0888620056211948e-003,0.4292691051959992,0.5301715731620789,0,2,0,8,16,2,-1.,0,9,16,1,2.,3.2383969519287348e-003,0.3792347908020020,0.5220744013786316,0,2,14,3,6,2,-1.,14,4,6,1,2.,4.9075027927756310e-003,0.5237283110618591,0.4126757979393005,0,2,0,0,5,6,-1.,0,2,5,2,3.,-0.0322779417037964,0.1947655975818634,0.4994502067565918,0,2,12,5,4,3,-1.,12,6,4,1,3.,-8.9711230248212814e-003,0.6011285185813904,0.4929032027721405,0,2,4,11,3,6,-1.,4,13,3,2,3.,0.0153210898861289,0.5009753704071045,0.2039822041988373,0,2,12,5,4,3,-1.,12,6,4,1,3.,2.0855569746345282e-003,0.4862189888954163,0.5721694827079773,0,2,9,5,1,3,-1.,9,6,1,1,3.,5.0615021027624607e-003,0.5000218749046326,0.1801805943250656,0,2,12,5,4,3,-1.,12,6,4,1,3.,-3.7174751050770283e-003,0.5530117154121399,0.4897592961788178,0,2,6,6,8,12,-1.,6,12,8,6,2.,-0.0121705001220107,0.4178605973720551,0.5383723974227905,0,2,12,5,4,3,-1.,12,6,4,1,3.,4.6248398721218109e-003,0.4997169971466065,0.5761327147483826,0,2,5,12,9,2,-1.,8,12,3,2,3.,-2.1040429419372231e-004,0.5331807136535645,0.4097681045532227,0,2,12,5,4,3,-1.,12,6,4,1,3.,-0.0146417804062366,0.5755925178527832,0.5051776170730591,0,2,4,5,4,3,-1.,4,6,4,1,3.,3.3199489116668701e-003,0.4576976895332336,0.6031805872917175,0,2,6,6,9,2,-1.,9,6,3,2,3.,3.7236879579722881e-003,0.4380396902561188,0.5415883064270020,0,2,4,11,1,3,-1.,4,12,1,1,3.,8.2951161311939359e-004,0.5163031816482544,0.3702219128608704,0,2,14,12,6,6,-1.,14,12,3,6,2.,-0.0114084901288152,0.6072946786880493,0.4862565100193024,0,2,7,0,3,7,-1.,8,0,1,7,3.,-4.5320121571421623e-003,0.3292475938796997,0.5088962912559509,0,2,9,8,3,3,-1.,10,8,1,3,3.,5.1276017911732197e-003,0.4829767942428589,0.6122708916664124,0,2,8,8,3,3,-1.,9,8,1,3,3.,9.8583158105611801e-003,0.4660679996013641,0.6556177139282227,0,2,5,10,11,3,-1.,5,11,11,1,3.,0.0369859188795090,0.5204849243164063,0.1690472066402435,0,2,5,7,10,1,-1.,10,7,5,1,2.,4.6491161920130253e-003,0.5167322158813477,0.3725225031375885,0,2,9,7,3,2,-1.,10,7,1,2,3.,-4.2664702050387859e-003,0.6406493186950684,0.4987342953681946,0,2,8,7,3,2,-1.,9,7,1,2,3.,-4.7956590424291790e-004,0.5897293090820313,0.4464873969554901,0,2,11,9,4,2,-1.,11,9,2,2,2.,3.6827160511165857e-003,0.5441560745239258,0.3472662866115570,0,2,5,9,4,2,-1.,7,9,2,2,2.,-0.0100598800927401,0.2143162935972214,0.5004829764366150,0,2,14,10,2,4,-1.,14,12,2,2,2.,-3.0361840617842972e-004,0.5386424064636231,0.4590323865413666,0,2,7,7,3,2,-1.,8,7,1,2,3.,-1.4545479789376259e-003,0.5751184225082398,0.4497095048427582,0,2,14,17,6,3,-1.,14,18,6,1,3.,1.6515209572389722e-003,0.5421937704086304,0.4238520860671997,0,3,4,5,12,12,-1.,4,5,6,6,2.,10,11,6,6,2.,-7.8468639403581619e-003,0.4077920913696289,0.5258157253265381,0,3,6,9,8,8,-1.,10,9,4,4,2.,6,13,4,4,2.,-5.1259850151836872e-003,0.4229275882244110,0.5479453206062317,0,2,0,4,15,4,-1.,5,4,5,4,3.,-0.0368909612298012,0.6596375703811646,0.4674678146839142,0,2,13,2,4,1,-1.,13,2,2,1,2.,2.4035639944486320e-004,0.4251135885715485,0.5573202967643738,0,2,4,12,2,2,-1.,4,13,2,1,2.,-1.5150169929256663e-005,0.5259246826171875,0.4074114859104157,0,2,8,13,4,3,-1.,8,14,4,1,3.,2.2108471021056175e-003,0.4671722948551178,0.5886352062225342,0,2,9,13,2,3,-1.,9,14,2,1,3.,-1.1568620102480054e-003,0.5711066126823425,0.4487161934375763,0,2,13,11,2,3,-1.,13,12,2,1,3.,4.9996292218565941e-003,0.5264198184013367,0.2898327112197876,0,3,7,12,4,4,-1.,7,12,2,2,2.,9,14,2,2,2.,-1.4656189596280456e-003,0.3891738057136536,0.5197871923446655,0,3,10,11,2,2,-1.,11,11,1,1,2.,10,12,1,1,2.,-1.1975039960816503e-003,0.5795872807502747,0.4927955865859985,0,2,8,17,3,2,-1.,9,17,1,2,3.,-4.4954330660402775e-003,0.2377603054046631,0.5012555122375488,0,3,10,11,2,2,-1.,11,11,1,1,2.,10,12,1,1,2.,1.4997160178609192e-004,0.4876626133918762,0.5617607831954956,0,2,0,17,6,3,-1.,0,18,6,1,3.,2.6391509454697371e-003,0.5168088078498840,0.3765509128570557,0,3,10,11,2,2,-1.,11,11,1,1,2.,10,12,1,1,2.,-2.9368131072260439e-004,0.5446649193763733,0.4874630868434906,0,3,8,11,2,2,-1.,8,11,1,1,2.,9,12,1,1,2.,1.4211760135367513e-003,0.4687897861003876,0.6691331863403320,0,2,12,5,8,4,-1.,12,5,4,4,2.,0.0794276371598244,0.5193443894386292,0.2732945978641510,0,2,0,5,8,4,-1.,4,5,4,4,2.,0.0799375027418137,0.4971731007099152,0.1782083958387375,0,2,13,2,4,1,-1.,13,2,2,1,2.,0.0110892597585917,0.5165994763374329,0.3209475874900818,0,2,3,2,4,1,-1.,5,2,2,1,2.,1.6560709627810866e-004,0.4058471918106079,0.5307276248931885,0,3,10,0,4,2,-1.,12,0,2,1,2.,10,1,2,1,2.,-5.3354292176663876e-003,0.3445056974887848,0.5158129930496216,0,2,7,12,3,1,-1.,8,12,1,1,3.,1.1287260567769408e-003,0.4594863057136536,0.6075533032417297,0,3,8,11,4,8,-1.,10,11,2,4,2.,8,15,2,4,2.,-0.0219692196696997,0.1680400967597961,0.5228595733642578,0,2,9,9,2,2,-1.,9,10,2,1,2.,-2.1775320055894554e-004,0.3861596882343292,0.5215672850608826,0,2,3,18,15,2,-1.,3,19,15,1,2.,2.0200149447191507e-004,0.5517979264259338,0.4363039135932922,0,3,2,6,2,12,-1.,2,6,1,6,2.,3,12,1,6,2.,-0.0217331498861313,0.7999460101127625,0.4789851009845734,0,2,9,8,2,3,-1.,9,9,2,1,3.,-8.4399932529777288e-004,0.4085975885391235,0.5374773144721985,0,2,7,10,3,2,-1.,8,10,1,2,3.,-4.3895249837078154e-004,0.5470405220985413,0.4366143047809601,0,2,11,11,3,1,-1.,12,11,1,1,3.,1.5092400135472417e-003,0.4988996982574463,0.5842149257659912,0,2,6,11,3,1,-1.,7,11,1,1,3.,-3.5547839943319559e-003,0.6753690242767334,0.4721005856990814,0,3,9,2,4,2,-1.,11,2,2,1,2.,9,3,2,1,2.,4.8191400128416717e-004,0.5415853857994080,0.4357109069824219,0,2,4,12,2,3,-1.,4,13,2,1,3.,-6.0264398343861103e-003,0.2258509993553162,0.4991880953311920,0,2,2,1,18,3,-1.,8,1,6,3,3.,-0.0116681400686502,0.6256554722785950,0.4927498996257782,0,2,5,1,4,14,-1.,7,1,2,14,2.,-2.8718370012938976e-003,0.3947784900665283,0.5245801806449890,0,2,8,16,12,3,-1.,8,16,6,3,2.,0.0170511696487665,0.4752511084079742,0.5794224143028259,0,2,1,17,18,3,-1.,7,17,6,3,3.,-0.0133520802482963,0.6041104793548584,0.4544535875320435,0,2,9,14,2,6,-1.,9,17,2,3,2.,-3.9301801007241011e-004,0.4258275926113129,0.5544905066490173,0,2,9,12,1,8,-1.,9,16,1,4,2.,3.0483349692076445e-003,0.5233420133590698,0.3780272901058197,0,2,9,14,2,3,-1.,9,15,2,1,3.,-4.3579288758337498e-003,0.6371889114379883,0.4838674068450928,0,2,9,6,2,12,-1.,9,10,2,4,3.,5.6661018170416355e-003,0.5374705791473389,0.4163666069507599,0,2,12,9,3,3,-1.,12,10,3,1,3.,6.0677339206449687e-005,0.4638795852661133,0.5311625003814697,0,2,0,1,4,8,-1.,2,1,2,8,2.,0.0367381609976292,0.4688656032085419,0.6466524004936218,0,3,9,1,6,2,-1.,12,1,3,1,2.,9,2,3,1,2.,8.6528137326240540e-003,0.5204318761825562,0.2188657969236374,0,2,1,3,12,14,-1.,1,10,12,7,2.,-0.1537135988473892,0.1630371958017349,0.4958840012550354,0,3,8,12,4,2,-1.,10,12,2,1,2.,8,13,2,1,2.,-4.1560421232134104e-004,0.5774459242820740,0.4696458876132965,0,3,1,9,10,2,-1.,1,9,5,1,2.,6,10,5,1,2.,-1.2640169588848948e-003,0.3977175951004028,0.5217198133468628,0,2,8,15,4,3,-1.,8,16,4,1,3.,-3.5473341122269630e-003,0.6046528220176697,0.4808315038681030,0,2,6,8,8,3,-1.,6,9,8,1,3.,3.0019069527043030e-005,0.3996723890304565,0.5228201150894165,0,2,9,15,5,3,-1.,9,16,5,1,3.,1.3113019522279501e-003,0.4712158143520355,0.5765997767448425,0,2,8,7,4,3,-1.,8,8,4,1,3.,-1.3374709524214268e-003,0.4109584987163544,0.5253170132637024,0,2,7,7,6,2,-1.,7,8,6,1,2.,0.0208767093718052,0.5202993750572205,0.1757981926202774,0,3,5,7,8,2,-1.,5,7,4,1,2.,9,8,4,1,2.,-7.5497948564589024e-003,0.6566609740257263,0.4694975018501282,0,2,12,9,3,3,-1.,12,10,3,1,3.,0.0241885501891375,0.5128673911094666,0.3370220959186554,0,2,4,7,4,2,-1.,4,8,4,1,2.,-2.9358828905969858e-003,0.6580786705017090,0.4694541096687317,0,2,14,2,6,9,-1.,14,5,6,3,3.,0.0575579293072224,0.5146445035934448,0.2775259912014008,0,2,4,9,3,3,-1.,5,9,1,3,3.,-1.1343370424583554e-003,0.3836601972579956,0.5192667245864868,0,2,12,9,3,3,-1.,12,10,3,1,3.,0.0168169997632504,0.5085592865943909,0.6177260875701904,0,2,0,2,6,9,-1.,0,5,6,3,3.,5.0535178743302822e-003,0.5138763189315796,0.3684791922569275,0,2,17,3,3,6,-1.,18,3,1,6,3.,-4.5874710194766521e-003,0.5989655256271362,0.4835202097892761,0,2,0,3,3,6,-1.,1,3,1,6,3.,1.6882460331544280e-003,0.4509486854076386,0.5723056793212891,0,2,17,14,1,2,-1.,17,15,1,1,2.,-1.6554000321775675e-003,0.3496770858764648,0.5243319272994995,0,2,4,9,4,3,-1.,6,9,2,3,2.,-0.0193738006055355,0.1120536997914314,0.4968712925910950,0,2,12,9,3,3,-1.,12,10,3,1,3.,0.0103744501248002,0.5148196816444397,0.4395213127136231,0,2,5,9,3,3,-1.,5,10,3,1,3.,1.4973050565458834e-004,0.4084999859333038,0.5269886851310730,0,3,9,5,6,8,-1.,12,5,3,4,2.,9,9,3,4,2.,-0.0429819300770760,0.6394104957580566,0.5018504261970520,0,3,5,5,6,8,-1.,5,5,3,4,2.,8,9,3,4,2.,8.3065936341881752e-003,0.4707553982734680,0.6698353290557861,0,2,16,1,4,6,-1.,16,4,4,3,2.,-4.1285790503025055e-003,0.4541369080543518,0.5323647260665894,0,2,1,0,6,20,-1.,3,0,2,20,3.,1.7399420030415058e-003,0.4333961904048920,0.5439866185188294,0,2,12,11,3,2,-1.,13,11,1,2,3.,1.1739750334527344e-004,0.4579687118530273,0.5543426275253296,0,2,5,11,3,2,-1.,6,11,1,2,3.,1.8585780344437808e-004,0.4324643909931183,0.5426754951477051,0,2,9,4,6,1,-1.,11,4,2,1,3.,5.5587692186236382e-003,0.5257220864295960,0.3550611138343811,0,2,0,0,8,3,-1.,4,0,4,3,2.,-7.9851560294628143e-003,0.6043018102645874,0.4630635976791382,0,2,15,0,2,5,-1.,15,0,1,5,2.,6.0594122624024749e-004,0.4598254859447479,0.5533195137977600,0,2,4,1,3,2,-1.,5,1,1,2,3.,-2.2983040253166109e-004,0.4130752086639404,0.5322461128234863,0,2,7,0,6,15,-1.,9,0,2,15,3.,4.3740210821852088e-004,0.4043039977550507,0.5409289002418518,0,2,6,11,3,1,-1.,7,11,1,1,3.,2.9482020181603730e-004,0.4494963884353638,0.5628852248191834,0,2,12,0,3,4,-1.,13,0,1,4,3.,0.0103126596659422,0.5177510976791382,0.2704316973686218,0,2,5,4,6,1,-1.,7,4,2,1,3.,-7.7241109684109688e-003,0.1988019049167633,0.4980553984642029,0,2,12,7,3,2,-1.,12,8,3,1,2.,-4.6797208487987518e-003,0.6644750237464905,0.5018296241760254,0,2,0,1,4,6,-1.,0,4,4,3,2.,-5.0755459815263748e-003,0.3898304998874664,0.5185269117355347,0,2,12,7,3,2,-1.,12,8,3,1,2.,2.2479740437120199e-003,0.4801808893680573,0.5660336017608643,0,2,2,16,3,3,-1.,2,17,3,1,3.,8.3327008178457618e-004,0.5210919976234436,0.3957188129425049,0,3,13,8,6,10,-1.,16,8,3,5,2.,13,13,3,5,2.,-0.0412793308496475,0.6154541969299316,0.5007054209709168,0,2,0,9,5,2,-1.,0,10,5,1,2.,-5.0930189900100231e-004,0.3975942134857178,0.5228403806686401,0,3,12,11,2,2,-1.,13,11,1,1,2.,12,12,1,1,2.,1.2568780221045017e-003,0.4979138076305389,0.5939183235168457,0,2,3,15,3,3,-1.,3,16,3,1,3.,8.0048497766256332e-003,0.4984497129917145,0.1633366048336029,0,2,12,7,3,2,-1.,12,8,3,1,2.,-1.1879300000146031e-003,0.5904964804649353,0.4942624866962433,0,2,5,7,3,2,-1.,5,8,3,1,2.,6.1948952497914433e-004,0.4199557900428772,0.5328726172447205,0,2,9,5,9,9,-1.,9,8,9,3,3.,6.6829859279096127e-003,0.5418602824211121,0.4905889034271240,0,2,5,0,3,7,-1.,6,0,1,7,3.,-3.7062340416014194e-003,0.3725939095020294,0.5138000249862671,0,2,5,2,12,5,-1.,9,2,4,5,3.,-0.0397394113242626,0.6478961110115051,0.5050346851348877,0,3,6,11,2,2,-1.,6,11,1,1,2.,7,12,1,1,2.,1.4085009461268783e-003,0.4682339131832123,0.6377884149551392,0,2,15,15,3,2,-1.,15,16,3,1,2.,3.9322688826359808e-004,0.5458530187606812,0.4150482118129730,0,2,2,15,3,2,-1.,2,16,3,1,2.,-1.8979819724336267e-003,0.3690159916877747,0.5149704217910767,0,3,14,12,6,8,-1.,17,12,3,4,2.,14,16,3,4,2.,-0.0139704402536154,0.6050562858581543,0.4811357855796814,0,2,2,8,15,6,-1.,7,8,5,6,3.,-0.1010081991553307,0.2017080038785934,0.4992361962795258,0,2,2,2,18,17,-1.,8,2,6,17,3.,-0.0173469204455614,0.5713148713111877,0.4899486005306244,0,2,5,1,4,1,-1.,7,1,2,1,2.,1.5619759506080300e-004,0.4215388894081116,0.5392642021179199,0,2,5,2,12,5,-1.,9,2,4,5,3.,0.1343892961740494,0.5136151909828186,0.3767612874507904,0,2,3,2,12,5,-1.,7,2,4,5,3.,-0.0245822407305241,0.7027357816696167,0.4747906923294067,0,3,4,9,12,4,-1.,10,9,6,2,2.,4,11,6,2,2.,-3.8553720805794001e-003,0.4317409098148346,0.5427716970443726,0,3,5,15,6,2,-1.,5,15,3,1,2.,8,16,3,1,2.,-2.3165249731391668e-003,0.5942698717117310,0.4618647992610931,0,2,10,14,2,3,-1.,10,15,2,1,3.,-4.8518120311200619e-003,0.6191568970680237,0.4884895086288452,0,3,0,13,20,2,-1.,0,13,10,1,2.,10,14,10,1,2.,2.4699938949197531e-003,0.5256664752960205,0.4017199873924255,0,3,4,9,12,8,-1.,10,9,6,4,2.,4,13,6,4,2.,0.0454969592392445,0.5237867832183838,0.2685773968696594,0,2,8,13,3,6,-1.,8,16,3,3,2.,-0.0203195996582508,0.2130445986986160,0.4979738891124725,0,2,10,12,2,2,-1.,10,13,2,1,2.,2.6994998916052282e-004,0.4814041852951050,0.5543122291564941,0,3,9,12,2,2,-1.,9,12,1,1,2.,10,13,1,1,2.,-1.8232699949294329e-003,0.6482579708099365,0.4709989130496979,0,3,4,11,14,4,-1.,11,11,7,2,2.,4,13,7,2,2.,-6.3015790656208992e-003,0.4581927955150604,0.5306236147880554,0,2,8,5,4,2,-1.,8,6,4,1,2.,-2.4139499873854220e-004,0.5232086777687073,0.4051763117313385,0,2,10,10,6,3,-1.,12,10,2,3,3.,-1.0330369696021080e-003,0.5556201934814453,0.4789193868637085,0,2,2,14,1,2,-1.,2,15,1,1,2.,1.8041160365100950e-004,0.5229442715644836,0.4011810123920441,0,3,13,8,6,12,-1.,16,8,3,6,2.,13,14,3,6,2.,-0.0614078603684902,0.6298682093620300,0.5010703206062317,0,3,1,8,6,12,-1.,1,8,3,6,2.,4,14,3,6,2.,-0.0695439130067825,0.7228280901908875,0.4773184061050415,0,2,10,0,6,10,-1.,12,0,2,10,3.,-0.0705426633358002,0.2269513010978699,0.5182529091835022,0,3,5,11,8,4,-1.,5,11,4,2,2.,9,13,4,2,2.,2.4423799477517605e-003,0.5237097144126892,0.4098151028156281,0,3,10,16,8,4,-1.,14,16,4,2,2.,10,18,4,2,2.,1.5494349645450711e-003,0.4773750901222229,0.5468043088912964,0,2,7,7,6,6,-1.,9,7,2,6,3.,-0.0239142198115587,0.7146975994110107,0.4783824980258942,0,2,10,2,4,10,-1.,10,2,2,10,2.,-0.0124536901712418,0.2635296881198883,0.5241122841835022,0,2,6,1,4,9,-1.,8,1,2,9,2.,-2.0760179904755205e-004,0.3623757064342499,0.5113608837127686,0,2,12,19,2,1,-1.,12,19,1,1,2.,2.9781080229440704e-005,0.4705932140350342,0.5432801842689514,104.7491989135742200,211,0,2,1,2,4,9,-1.,3,2,2,9,2.,0.0117727499455214,0.3860518932342529,0.6421167254447937,0,2,7,5,6,4,-1.,9,5,2,4,3.,0.0270375702530146,0.4385654926300049,0.6754038929939270,0,2,9,4,2,4,-1.,9,6,2,2,2.,-3.6419500247575343e-005,0.5487101078033447,0.3423315882682800,0,2,14,5,2,8,-1.,14,9,2,4,2.,1.9995409529656172e-003,0.3230532109737396,0.5400317907333374,0,2,7,6,5,12,-1.,7,12,5,6,2.,4.5278300531208515e-003,0.5091639757156372,0.2935043871402741,0,2,14,6,2,6,-1.,14,9,2,3,2.,4.7890920541249216e-004,0.4178153872489929,0.5344064235687256,0,2,4,6,2,6,-1.,4,9,2,3,2.,1.1720920447260141e-003,0.2899182140827179,0.5132070779800415,0,3,8,15,10,4,-1.,13,15,5,2,2.,8,17,5,2,2.,9.5305702416226268e-004,0.4280124902725220,0.5560845136642456,0,2,6,18,2,2,-1.,7,18,1,2,2.,1.5099150004971307e-005,0.4044871926307678,0.5404760241508484,0,2,11,3,6,2,-1.,11,4,6,1,2.,-6.0817901976406574e-004,0.4271768927574158,0.5503466129302979,0,2,2,0,16,6,-1.,2,2,16,2,3.,3.3224520739167929e-003,0.3962723910808563,0.5369734764099121,0,2,11,3,6,2,-1.,11,4,6,1,2.,-1.1037490330636501e-003,0.4727177917957306,0.5237749814987183,0,2,4,11,10,3,-1.,4,12,10,1,3.,-1.4350269921123981e-003,0.5603008270263672,0.4223509132862091,0,2,11,3,6,2,-1.,11,4,6,1,2.,2.0767399109899998e-003,0.5225917100906372,0.4732725918292999,0,2,3,3,6,2,-1.,3,4,6,1,2.,-1.6412809782195836e-004,0.3999075889587402,0.5432739853858948,0,2,16,0,4,7,-1.,16,0,2,7,2.,8.8302437216043472e-003,0.4678385853767395,0.6027327179908752,0,2,0,14,9,6,-1.,0,16,9,2,3.,-0.0105520701035857,0.3493967056274414,0.5213974714279175,0,2,9,16,3,3,-1.,9,17,3,1,3.,-2.2731600329279900e-003,0.6185818910598755,0.4749062955379486,0,2,4,6,6,2,-1.,6,6,2,2,3.,-8.4786332445219159e-004,0.5285341143608093,0.3843482136726379,0,2,15,11,1,3,-1.,15,12,1,1,3.,1.2081359745934606e-003,0.5360640883445740,0.3447335958480835,0,2,5,5,2,3,-1.,5,6,2,1,3.,2.6512730401009321e-003,0.4558292031288147,0.6193962097167969,0,2,10,9,2,2,-1.,10,10,2,1,2.,-1.1012479662895203e-003,0.3680230081081390,0.5327628254890442,0,2,3,1,4,3,-1.,5,1,2,3,2.,4.9561518244445324e-004,0.3960595130920410,0.5274940729141235,0,2,16,0,4,7,-1.,16,0,2,7,2.,-0.0439017713069916,0.7020444869995117,0.4992839097976685,0,2,0,0,20,1,-1.,10,0,10,1,2.,0.0346903502941132,0.5049164295196533,0.2766602933406830,0,2,15,11,1,3,-1.,15,12,1,1,3.,-2.7442190330475569e-003,0.2672632932662964,0.5274971127510071,0,2,0,4,3,4,-1.,1,4,1,4,3.,3.3316588960587978e-003,0.4579482972621918,0.6001101732254028,0,2,16,3,3,6,-1.,16,5,3,2,3.,-0.0200445707887411,0.3171594142913818,0.5235717892646790,0,2,1,3,3,6,-1.,1,5,3,2,3.,1.3492030557245016e-003,0.5265362858772278,0.4034324884414673,0,3,6,2,12,6,-1.,12,2,6,3,2.,6,5,6,3,2.,2.9702018946409225e-003,0.5332456827163696,0.4571984112262726,0,2,8,10,4,3,-1.,8,11,4,1,3.,6.3039981760084629e-003,0.4593310952186585,0.6034635901451111,0,3,4,2,14,6,-1.,11,2,7,3,2.,4,5,7,3,2.,-0.0129365902394056,0.4437963962554932,0.5372971296310425,0,2,9,11,2,3,-1.,9,12,2,1,3.,4.0148729458451271e-003,0.4680323898792267,0.6437833905220032,0,2,15,13,2,3,-1.,15,14,2,1,3.,-2.6401679497212172e-003,0.3709631860256195,0.5314332842826843,0,2,8,12,4,3,-1.,8,13,4,1,3.,0.0139184398576617,0.4723555147647858,0.7130808830261231,0,2,15,11,1,3,-1.,15,12,1,1,3.,-4.5087869511917233e-004,0.4492394030094147,0.5370404124259949,0,2,7,13,5,2,-1.,7,14,5,1,2.,2.5384349282830954e-004,0.4406864047050476,0.5514402985572815,0,2,7,12,6,3,-1.,7,13,6,1,3.,2.2710000630468130e-003,0.4682416915893555,0.5967984199523926,0,2,5,11,4,4,-1.,5,13,4,2,2.,2.4120779708027840e-003,0.5079392194747925,0.3018598854541779,0,2,11,4,3,3,-1.,12,4,1,3,3.,-3.6025670851813629e-005,0.5601037144660950,0.4471096992492676,0,2,6,4,3,3,-1.,7,4,1,3,3.,-7.4905529618263245e-003,0.2207535058259964,0.4989944100379944,0,2,16,5,3,6,-1.,17,5,1,6,3.,-0.0175131205469370,0.6531215906143189,0.5017648935317993,0,2,3,6,12,7,-1.,7,6,4,7,3.,0.1428163051605225,0.4967963099479675,0.1482062041759491,0,2,16,5,3,6,-1.,17,5,1,6,3.,5.5345268920063972e-003,0.4898946881294251,0.5954223871231079,0,2,3,13,2,3,-1.,3,14,2,1,3.,-9.6323591424152255e-004,0.3927116990089417,0.5196074247360230,0,2,16,5,3,6,-1.,17,5,1,6,3.,-2.0370010752230883e-003,0.5613325238227844,0.4884858131408691,0,2,1,5,3,6,-1.,2,5,1,6,3.,1.6614829655736685e-003,0.4472880065441132,0.5578880906105042,0,2,1,9,18,1,-1.,7,9,6,1,3.,-3.1188090797513723e-003,0.3840532898902893,0.5397477746009827,0,2,0,9,8,7,-1.,4,9,4,7,2.,-6.4000617712736130e-003,0.5843983888626099,0.4533218145370483,0,2,12,11,8,2,-1.,12,12,8,1,2.,3.1319601112045348e-004,0.5439221858978272,0.4234727919101715,0,2,0,11,8,2,-1.,0,12,8,1,2.,-0.0182220991700888,0.1288464963436127,0.4958404898643494,0,2,9,13,2,3,-1.,9,14,2,1,3.,8.7969247251749039e-003,0.4951297938823700,0.7153480052947998,0,3,4,10,12,4,-1.,4,10,6,2,2.,10,12,6,2,2.,-4.2395070195198059e-003,0.3946599960327148,0.5194936990737915,0,2,9,3,3,7,-1.,10,3,1,7,3.,9.7086271271109581e-003,0.4897503852844238,0.6064900159835815,0,2,7,2,3,5,-1.,8,2,1,5,3.,-3.9934171363711357e-003,0.3245440125465393,0.5060828924179077,0,3,9,12,4,6,-1.,11,12,2,3,2.,9,15,2,3,2.,-0.0167850591242313,0.1581953018903732,0.5203778743743897,0,2,8,7,3,6,-1.,9,7,1,6,3.,0.0182720907032490,0.4680935144424439,0.6626979112625122,0,2,15,4,4,2,-1.,15,5,4,1,2.,5.6872838176786900e-003,0.5211697816848755,0.3512184917926788,0,2,8,7,3,3,-1.,9,7,1,3,3.,-1.0739039862528443e-003,0.5768386125564575,0.4529845118522644,0,2,14,2,6,4,-1.,14,4,6,2,2.,-3.7093870341777802e-003,0.4507763087749481,0.5313581228256226,0,2,7,16,6,1,-1.,9,16,2,1,3.,-2.1110709349159151e-004,0.5460820198059082,0.4333376884460449,0,2,15,13,2,3,-1.,15,14,2,1,3.,1.0670139454305172e-003,0.5371856093406677,0.4078390896320343,0,2,8,7,3,10,-1.,9,7,1,10,3.,3.5943021066486835e-003,0.4471287131309509,0.5643836259841919,0,2,11,10,2,6,-1.,11,12,2,2,3.,-5.1776031032204628e-003,0.4499393105506897,0.5280330181121826,0,2,6,10,4,1,-1.,8,10,2,1,2.,-2.5414369883947074e-004,0.5516173243522644,0.4407708048820496,0,2,10,9,2,2,-1.,10,10,2,1,2.,6.3522560521960258e-003,0.5194190144538879,0.2465227991342545,0,2,8,9,2,2,-1.,8,10,2,1,2.,-4.4205080484971404e-004,0.3830705881118774,0.5139682292938232,0,3,12,7,2,2,-1.,13,7,1,1,2.,12,8,1,1,2.,7.4488727841526270e-004,0.4891090989112854,0.5974786877632141,0,3,5,7,2,2,-1.,5,7,1,1,2.,6,8,1,1,2.,-3.5116379149258137e-003,0.7413681745529175,0.4768764972686768,0,2,13,0,3,14,-1.,14,0,1,14,3.,-0.0125409103929996,0.3648819029331207,0.5252826809883118,0,2,4,0,3,14,-1.,5,0,1,14,3.,9.4931852072477341e-003,0.5100492835044861,0.3629586994647980,0,2,13,4,3,14,-1.,14,4,1,14,3.,0.0129611501470208,0.5232442021369934,0.4333561062812805,0,2,9,14,2,3,-1.,9,15,2,1,3.,4.7209449112415314e-003,0.4648149013519287,0.6331052780151367,0,2,8,14,4,3,-1.,8,15,4,1,3.,-2.3119079414755106e-003,0.5930309891700745,0.4531058073043823,0,2,4,2,3,16,-1.,5,2,1,16,3.,-2.8262299019843340e-003,0.3870477974414825,0.5257101058959961,0,2,7,2,8,10,-1.,7,7,8,5,2.,-1.4311339473351836e-003,0.5522503256797791,0.4561854898929596,0,2,6,14,7,3,-1.,6,15,7,1,3.,1.9378310535103083e-003,0.4546220898628235,0.5736966729164124,0,3,9,2,10,12,-1.,14,2,5,6,2.,9,8,5,6,2.,2.6343559147790074e-004,0.5345739126205444,0.4571875035762787,0,2,6,7,8,2,-1.,6,8,8,1,2.,7.8257522545754910e-004,0.3967815935611725,0.5220187902450562,0,2,8,13,4,6,-1.,8,16,4,3,2.,-0.0195504408329725,0.2829642891883850,0.5243508219718933,0,2,6,6,1,3,-1.,6,7,1,1,3.,4.3914958951063454e-004,0.4590066969394684,0.5899090170860291,0,2,16,2,4,6,-1.,16,4,4,2,3.,0.0214520003646612,0.5231410861015320,0.2855378985404968,0,3,6,6,4,2,-1.,6,6,2,1,2.,8,7,2,1,2.,5.8973580598831177e-004,0.4397256970405579,0.5506421923637390,0,2,16,2,4,6,-1.,16,4,4,2,3.,-0.0261576101183891,0.3135079145431519,0.5189175009727478,0,2,0,2,4,6,-1.,0,4,4,2,3.,-0.0139598604291677,0.3213272988796234,0.5040717720985413,0,2,9,6,2,6,-1.,9,6,1,6,2.,-6.3699018210172653e-003,0.6387544870376587,0.4849506914615631,0,2,3,4,6,10,-1.,3,9,6,5,2.,-8.5613820701837540e-003,0.2759132087230682,0.5032019019126892,0,2,9,5,2,6,-1.,9,5,1,6,2.,9.6622901037335396e-004,0.4685640931129456,0.5834879279136658,0,2,3,13,2,3,-1.,3,14,2,1,3.,7.6550268568098545e-004,0.5175207257270813,0.3896422088146210,0,2,13,13,3,2,-1.,13,14,3,1,2.,-8.1833340227603912e-003,0.2069136947393417,0.5208122134208679,0,3,2,16,10,4,-1.,2,16,5,2,2.,7,18,5,2,2.,-9.3976939097046852e-003,0.6134091019630432,0.4641222953796387,0,3,5,6,10,6,-1.,10,6,5,3,2.,5,9,5,3,2.,4.8028980381786823e-003,0.5454108119010925,0.4395219981670380,0,2,7,14,1,3,-1.,7,15,1,1,3.,-3.5680569708347321e-003,0.6344485282897949,0.4681093990802765,0,2,14,16,6,3,-1.,14,17,6,1,3.,4.0733120404183865e-003,0.5292683243751526,0.4015620052814484,0,2,5,4,3,3,-1.,5,5,3,1,3.,1.2568129459396005e-003,0.4392988085746765,0.5452824831008911,0,2,7,4,10,3,-1.,7,5,10,1,3.,-2.9065010603517294e-003,0.5898832082748413,0.4863379895687103,0,2,0,4,5,4,-1.,0,6,5,2,2.,-2.4409340694546700e-003,0.4069364964962006,0.5247421860694885,0,2,13,11,3,9,-1.,13,14,3,3,3.,0.0248307008296251,0.5182725787162781,0.3682524859905243,0,2,4,11,3,9,-1.,4,14,3,3,3.,-0.0488540083169937,0.1307577937841415,0.4961281120777130,0,2,9,7,2,1,-1.,9,7,1,1,2.,-1.6110379947349429e-003,0.6421005725860596,0.4872662127017975,0,2,5,0,6,17,-1.,7,0,2,17,3.,-0.0970094799995422,0.0477693490684032,0.4950988888740540,0,2,10,3,6,3,-1.,10,3,3,3,2.,1.1209240183234215e-003,0.4616267085075378,0.5354745984077454,0,2,2,2,15,4,-1.,7,2,5,4,3.,-1.3064090162515640e-003,0.6261854171752930,0.4638805985450745,0,3,8,2,8,2,-1.,12,2,4,1,2.,8,3,4,1,2.,4.5771620352752507e-004,0.5384417772293091,0.4646640121936798,0,2,8,1,3,6,-1.,8,3,3,2,3.,-6.3149951165542006e-004,0.3804047107696533,0.5130257010459900,0,2,9,17,2,2,-1.,9,18,2,1,2.,1.4505970466416329e-004,0.4554310142993927,0.5664461851119995,0,2,0,0,2,14,-1.,1,0,1,14,2.,-0.0164745505899191,0.6596958041191101,0.4715859889984131,0,2,12,0,7,3,-1.,12,1,7,1,3.,0.0133695797994733,0.5195466279983521,0.3035964965820313,0,2,1,14,1,2,-1.,1,15,1,1,2.,1.0271780047332868e-004,0.5229176282882690,0.4107066094875336,0,3,14,12,2,8,-1.,15,12,1,4,2.,14,16,1,4,2.,-5.5311559699475765e-003,0.6352887749671936,0.4960907101631165,0,2,1,0,7,3,-1.,1,1,7,1,3.,-2.6187049224972725e-003,0.3824546039104462,0.5140984058380127,0,3,14,12,2,8,-1.,15,12,1,4,2.,14,16,1,4,2.,5.0834268331527710e-003,0.4950439929962158,0.6220818758010864,0,3,6,0,8,12,-1.,6,0,4,6,2.,10,6,4,6,2.,0.0798181593418121,0.4952335953712463,0.1322475969791412,0,2,6,1,8,9,-1.,6,4,8,3,3.,-0.0992265865206718,0.7542728781700134,0.5008416771888733,0,2,5,2,2,2,-1.,5,3,2,1,2.,-6.5174017800018191e-004,0.3699302971363068,0.5130121111869812,0,3,13,14,6,6,-1.,16,14,3,3,2.,13,17,3,3,2.,-0.0189968496561050,0.6689178943634033,0.4921202957630158,0,3,0,17,20,2,-1.,0,17,10,1,2.,10,18,10,1,2.,0.0173468999564648,0.4983300864696503,0.1859198063611984,0,3,10,3,2,6,-1.,11,3,1,3,2.,10,6,1,3,2.,5.5082101607695222e-004,0.4574424028396606,0.5522121787071228,0,2,5,12,6,2,-1.,8,12,3,2,2.,2.0056050270795822e-003,0.5131744742393494,0.3856469988822937,0,2,10,7,6,13,-1.,10,7,3,13,2.,-7.7688191086053848e-003,0.4361700117588043,0.5434309244155884,0,2,5,15,10,5,-1.,10,15,5,5,2.,0.0508782789111137,0.4682720899581909,0.6840639710426331,0,2,10,4,4,10,-1.,10,4,2,10,2.,-2.2901780903339386e-003,0.4329245090484619,0.5306099057197571,0,2,5,7,2,1,-1.,6,7,1,1,2.,-1.5715380141045898e-004,0.5370057225227356,0.4378164112567902,0,2,10,3,6,7,-1.,10,3,3,7,2.,0.1051924005150795,0.5137274265289307,0.0673614665865898,0,2,4,3,6,7,-1.,7,3,3,7,2.,2.7198919560760260e-003,0.4112060964107513,0.5255665183067322,0,2,1,7,18,5,-1.,7,7,6,5,3.,0.0483377799391747,0.5404623746871948,0.4438967108726502,0,2,3,17,4,3,-1.,5,17,2,3,2.,9.5703761326149106e-004,0.4355969130992889,0.5399510860443115,0,3,8,14,12,6,-1.,14,14,6,3,2.,8,17,6,3,2.,-0.0253712590783834,0.5995175242424011,0.5031024813652039,0,3,0,13,20,4,-1.,0,13,10,2,2.,10,15,10,2,2.,0.0524579510092735,0.4950287938117981,0.1398351043462753,0,3,4,5,14,2,-1.,11,5,7,1,2.,4,6,7,1,2.,-0.0123656298965216,0.6397299170494080,0.4964106082916260,0,3,1,2,10,12,-1.,1,2,5,6,2.,6,8,5,6,2.,-0.1458971947431564,0.1001669988036156,0.4946322143077850,0,2,6,1,14,3,-1.,6,2,14,1,3.,-0.0159086007624865,0.3312329947948456,0.5208340883255005,0,2,8,16,2,3,-1.,8,17,2,1,3.,3.9486068999394774e-004,0.4406363964080811,0.5426102876663208,0,2,9,17,3,2,-1.,10,17,1,2,3.,-5.2454001270234585e-003,0.2799589931964874,0.5189967155456543,0,3,5,15,4,2,-1.,5,15,2,1,2.,7,16,2,1,2.,-5.0421799533069134e-003,0.6987580060958862,0.4752142131328583,0,2,10,15,1,3,-1.,10,16,1,1,3.,2.9812189750373363e-003,0.4983288943767548,0.6307479739189148,0,3,8,16,4,4,-1.,8,16,2,2,2.,10,18,2,2,2.,-7.2884308174252510e-003,0.2982333004474640,0.5026869773864746,0,2,6,11,8,6,-1.,6,14,8,3,2.,1.5094350092113018e-003,0.5308442115783691,0.3832970857620239,0,2,2,13,5,2,-1.,2,14,5,1,2.,-9.3340799212455750e-003,0.2037964016199112,0.4969817101955414,0,3,13,14,6,6,-1.,16,14,3,3,2.,13,17,3,3,2.,0.0286671407520771,0.5025696754455566,0.6928027272224426,0,2,1,9,18,4,-1.,7,9,6,4,3.,0.1701968014240265,0.4960052967071533,0.1476442962884903,0,3,13,14,6,6,-1.,16,14,3,3,2.,13,17,3,3,2.,-3.2614478841423988e-003,0.5603063702583313,0.4826056063175201,0,2,0,2,1,6,-1.,0,4,1,2,3.,5.5769277969375253e-004,0.5205562114715576,0.4129633009433746,0,2,5,0,15,20,-1.,5,10,15,10,2.,0.3625833988189697,0.5221652984619141,0.3768612146377564,0,3,1,14,6,6,-1.,1,14,3,3,2.,4,17,3,3,2.,-0.0116151301190257,0.6022682785987854,0.4637489914894104,0,3,8,14,4,6,-1.,10,14,2,3,2.,8,17,2,3,2.,-4.0795197710394859e-003,0.4070447087287903,0.5337479114532471,0,2,7,11,2,1,-1.,8,11,1,1,2.,5.7204300537705421e-004,0.4601835012435913,0.5900393128395081,0,2,9,17,3,2,-1.,10,17,1,2,3.,6.7543348995968699e-004,0.5398252010345459,0.4345428943634033,0,2,8,17,3,2,-1.,9,17,1,2,3.,6.3295697327703238e-004,0.5201563239097595,0.4051358997821808,0,3,12,14,4,6,-1.,14,14,2,3,2.,12,17,2,3,2.,1.2435320531949401e-003,0.4642387926578522,0.5547441244125366,0,3,4,14,4,6,-1.,4,14,2,3,2.,6,17,2,3,2.,-4.7363857738673687e-003,0.6198567152023315,0.4672552049160004,0,3,13,14,2,6,-1.,14,14,1,3,2.,13,17,1,3,2.,-6.4658462069928646e-003,0.6837332844734192,0.5019000768661499,0,3,5,14,2,6,-1.,5,14,1,3,2.,6,17,1,3,2.,3.5017321351915598e-004,0.4344803094863892,0.5363622903823853,0,2,7,0,6,12,-1.,7,4,6,4,3.,1.5754920605104417e-004,0.4760079085826874,0.5732020735740662,0,2,0,7,12,2,-1.,4,7,4,2,3.,9.9774366244673729e-003,0.5090985894203186,0.3635039925575256,0,2,10,3,3,13,-1.,11,3,1,13,3.,-4.1464529931545258e-004,0.5570064783096314,0.4593802094459534,0,2,7,3,3,13,-1.,8,3,1,13,3.,-3.5888899583369493e-004,0.5356845855712891,0.4339134991168976,0,2,10,8,6,3,-1.,10,9,6,1,3.,4.0463250479660928e-004,0.4439803063869476,0.5436776876449585,0,2,3,11,3,2,-1.,4,11,1,2,3.,-8.2184787606820464e-004,0.4042294919490814,0.5176299214363098,0,3,13,12,6,8,-1.,16,12,3,4,2.,13,16,3,4,2.,5.9467419050633907e-003,0.4927651882171631,0.5633779764175415,0,2,7,6,6,5,-1.,9,6,2,5,3.,-0.0217533893883228,0.8006293773651123,0.4800840914249420,0,2,17,11,2,7,-1.,17,11,1,7,2.,-0.0145403798669577,0.3946054875850678,0.5182222723960877,0,2,3,13,8,2,-1.,7,13,4,2,2.,-0.0405107699334621,0.0213249903172255,0.4935792982578278,0,2,6,9,8,3,-1.,6,10,8,1,3.,-5.8458268176764250e-004,0.4012795984745026,0.5314025282859802,0,2,4,3,4,3,-1.,4,4,4,1,3.,5.5151800625026226e-003,0.4642418920993805,0.5896260738372803,0,2,11,3,4,3,-1.,11,4,4,1,3.,-6.0626221820712090e-003,0.6502159237861633,0.5016477704048157,0,2,1,4,17,12,-1.,1,8,17,4,3.,0.0945358425378799,0.5264708995819092,0.4126827120780945,0,2,11,3,4,3,-1.,11,4,4,1,3.,4.7315051779150963e-003,0.4879199862480164,0.5892447829246521,0,2,4,8,6,3,-1.,4,9,6,1,3.,-5.2571471314877272e-004,0.3917280137538910,0.5189412832260132,0,2,12,3,5,3,-1.,12,4,5,1,3.,-2.5464049540460110e-003,0.5837599039077759,0.4985705912113190,0,2,1,11,2,7,-1.,2,11,1,7,2.,-0.0260756891220808,0.1261983960866928,0.4955821931362152,0,3,15,12,2,8,-1.,16,12,1,4,2.,15,16,1,4,2.,-5.4779709316790104e-003,0.5722513794898987,0.5010265707969666,0,2,4,8,11,3,-1.,4,9,11,1,3.,5.1337741315364838e-003,0.5273262262344360,0.4226376116275787,0,3,9,13,6,2,-1.,12,13,3,1,2.,9,14,3,1,2.,4.7944980906322598e-004,0.4450066983699799,0.5819587111473084,0,2,6,13,4,3,-1.,6,14,4,1,3.,-2.1114079281687737e-003,0.5757653117179871,0.4511714875698090,0,2,9,12,3,3,-1.,10,12,1,3,3.,-0.0131799904629588,0.1884381026029587,0.5160734057426453,0,2,5,3,3,3,-1.,5,4,3,1,3.,-4.7968099825084209e-003,0.6589789986610413,0.4736118912696838,0,2,9,4,2,3,-1.,9,5,2,1,3.,6.7483168095350266e-003,0.5259429812431335,0.3356395065784454,0,2,0,2,16,3,-1.,0,3,16,1,3.,1.4623369788751006e-003,0.5355271100997925,0.4264092147350311,0,3,15,12,2,8,-1.,16,12,1,4,2.,15,16,1,4,2.,4.7645159065723419e-003,0.5034406781196594,0.5786827802658081,0,3,3,12,2,8,-1.,3,12,1,4,2.,4,16,1,4,2.,6.8066660314798355e-003,0.4756605029106140,0.6677829027175903,0,2,14,13,3,6,-1.,14,15,3,2,3.,3.6608621012419462e-003,0.5369611978530884,0.4311546981334686,0,2,3,13,3,6,-1.,3,15,3,2,3.,0.0214496403932571,0.4968641996383667,0.1888816058635712,0,3,6,5,10,2,-1.,11,5,5,1,2.,6,6,5,1,2.,4.1678901761770248e-003,0.4930733144283295,0.5815368890762329,0,2,2,14,14,6,-1.,2,17,14,3,2.,8.6467564105987549e-003,0.5205205082893372,0.4132595062255859,0,2,10,14,1,3,-1.,10,15,1,1,3.,-3.6114078829996288e-004,0.5483555197715759,0.4800927937030792,0,3,4,16,2,2,-1.,4,16,1,1,2.,5,17,1,1,2.,1.0808729566633701e-003,0.4689902067184448,0.6041421294212341,0,2,10,6,2,3,-1.,10,7,2,1,3.,5.7719959877431393e-003,0.5171142220497131,0.3053277134895325,0,3,0,17,20,2,-1.,0,17,10,1,2.,10,18,10,1,2.,1.5720770461484790e-003,0.5219978094100952,0.4178803861141205,0,2,13,6,1,3,-1.,13,7,1,1,3.,-1.9307859474793077e-003,0.5860369801521301,0.4812920093536377,0,2,8,13,3,2,-1.,9,13,1,2,3.,-7.8926272690296173e-003,0.1749276965856552,0.4971733987331390,0,2,12,2,3,3,-1.,13,2,1,3,3.,-2.2224679123610258e-003,0.4342589080333710,0.5212848186492920,0,3,3,18,2,2,-1.,3,18,1,1,2.,4,19,1,1,2.,1.9011989934369922e-003,0.4765186905860901,0.6892055273056030,0,2,9,16,3,4,-1.,10,16,1,4,3.,2.7576119173318148e-003,0.5262191295623779,0.4337486028671265,0,2,6,6,1,3,-1.,6,7,1,1,3.,5.1787449046969414e-003,0.4804069101810455,0.7843729257583618,0,2,13,1,5,2,-1.,13,2,5,1,2.,-9.0273341629654169e-004,0.4120846986770630,0.5353423953056335,0,3,7,14,6,2,-1.,7,14,3,1,2.,10,15,3,1,2.,5.1797959022223949e-003,0.4740372896194458,0.6425960063934326,0,2,11,3,3,4,-1.,12,3,1,4,3.,-0.0101140001788735,0.2468792051076889,0.5175017714500427,0,2,1,13,12,6,-1.,5,13,4,6,3.,-0.0186170600354671,0.5756294131278992,0.4628978967666626,0,2,14,11,5,2,-1.,14,12,5,1,2.,5.9225959703326225e-003,0.5169625878334045,0.3214271068572998,0,3,2,15,14,4,-1.,2,15,7,2,2.,9,17,7,2,2.,-6.2945079989731312e-003,0.3872014880180359,0.5141636729240418,0,3,3,7,14,2,-1.,10,7,7,1,2.,3,8,7,1,2.,6.5353019163012505e-003,0.4853048920631409,0.6310489773750305,0,2,1,11,4,2,-1.,1,12,4,1,2.,1.0878399480134249e-003,0.5117315053939819,0.3723258972167969,0,2,14,0,6,14,-1.,16,0,2,14,3.,-0.0225422400981188,0.5692740082740784,0.4887112975120544,0,2,4,11,1,3,-1.,4,12,1,1,3.,-3.0065660830587149e-003,0.2556012868881226,0.5003992915153503,0,2,14,0,6,14,-1.,16,0,2,14,3.,7.4741272255778313e-003,0.4810872972011566,0.5675926804542542,0,2,1,10,3,7,-1.,2,10,1,7,3.,0.0261623207479715,0.4971194863319397,0.1777237057685852,0,2,8,12,9,2,-1.,8,13,9,1,2.,9.4352738233283162e-004,0.4940010905265808,0.5491250753402710,0,2,0,6,20,1,-1.,10,6,10,1,2.,0.0333632417023182,0.5007612109184265,0.2790724039077759,0,2,8,4,4,4,-1.,8,4,2,4,2.,-0.0151186501607299,0.7059578895568848,0.4973031878471375,0,2,0,0,2,2,-1.,0,1,2,1,2.,9.8648946732282639e-004,0.5128620266914368,0.3776761889457703,105.7611007690429700,213,0,2,5,3,10,9,-1.,5,6,10,3,3.,-0.0951507985591888,0.6470757126808167,0.4017286896705627,0,2,15,2,4,10,-1.,15,2,2,10,2.,6.2702340073883533e-003,0.3999822139739990,0.5746449232101440,0,2,8,2,2,7,-1.,9,2,1,7,2.,3.0018089455552399e-004,0.3558770120143890,0.5538809895515442,0,2,7,4,12,1,-1.,11,4,4,1,3.,1.1757409665733576e-003,0.4256534874439240,0.5382617712020874,0,2,3,4,9,1,-1.,6,4,3,1,3.,4.4235268433112651e-005,0.3682908117771149,0.5589926838874817,0,2,15,10,1,4,-1.,15,12,1,2,2.,-2.9936920327600092e-005,0.5452470183372498,0.4020367860794067,0,2,4,10,6,4,-1.,7,10,3,4,2.,3.0073199886828661e-003,0.5239058136940002,0.3317843973636627,0,2,15,9,1,6,-1.,15,12,1,3,2.,-0.0105138896033168,0.4320689141750336,0.5307983756065369,0,2,7,17,6,3,-1.,7,18,6,1,3.,8.3476826548576355e-003,0.4504637122154236,0.6453298926353455,0,3,14,3,2,16,-1.,15,3,1,8,2.,14,11,1,8,2.,-3.1492270063608885e-003,0.4313425123691559,0.5370525121688843,0,2,4,9,1,6,-1.,4,12,1,3,2.,-1.4435649973165710e-005,0.5326603055000305,0.3817971944808960,0,2,12,1,5,2,-1.,12,2,5,1,2.,-4.2855090578086674e-004,0.4305163919925690,0.5382009744644165,0,3,6,18,4,2,-1.,6,18,2,1,2.,8,19,2,1,2.,1.5062429883982986e-004,0.4235970973968506,0.5544965267181397,0,3,2,4,16,10,-1.,10,4,8,5,2.,2,9,8,5,2.,0.0715598315000534,0.5303059816360474,0.2678802907466888,0,2,6,5,1,10,-1.,6,10,1,5,2.,8.4095180500298738e-004,0.3557108938694000,0.5205433964729309,0,2,4,8,15,2,-1.,9,8,5,2,3.,0.0629865005612373,0.5225362777709961,0.2861376106739044,0,2,1,8,15,2,-1.,6,8,5,2,3.,-3.3798629883676767e-003,0.3624185919761658,0.5201697945594788,0,2,9,5,3,6,-1.,9,7,3,2,3.,-1.1810739670181647e-004,0.5474476814270020,0.3959893882274628,0,2,5,7,8,2,-1.,9,7,4,2,2.,-5.4505601292476058e-004,0.3740422129631043,0.5215715765953064,0,2,9,11,2,3,-1.,9,12,2,1,3.,-1.8454910023137927e-003,0.5893052220344544,0.4584448933601379,0,2,1,0,16,3,-1.,1,1,16,1,3.,-4.3832371011376381e-004,0.4084582030773163,0.5385351181030273,0,2,11,2,7,2,-1.,11,3,7,1,2.,-2.4000830017030239e-003,0.3777455091476440,0.5293580293655396,0,2,5,1,10,18,-1.,5,7,10,6,3.,-0.0987957417964935,0.2963612079620361,0.5070089101791382,0,2,17,4,3,2,-1.,18,4,1,2,3.,3.1798239797353745e-003,0.4877632856369019,0.6726443767547607,0,2,8,13,1,3,-1.,8,14,1,1,3.,3.2406419632025063e-004,0.4366911053657532,0.5561109781265259,0,2,3,14,14,6,-1.,3,16,14,2,3.,-0.0325472503900528,0.3128157854080200,0.5308616161346436,0,2,0,2,3,4,-1.,1,2,1,4,3.,-7.7561130747199059e-003,0.6560224890708923,0.4639872014522553,0,2,12,1,5,2,-1.,12,2,5,1,2.,0.0160272493958473,0.5172680020332336,0.3141897916793823,0,2,3,1,5,2,-1.,3,2,5,1,2.,7.1002350523485802e-006,0.4084446132183075,0.5336294770240784,0,2,10,13,2,3,-1.,10,14,2,1,3.,7.3422808200120926e-003,0.4966922104358673,0.6603465080261231,0,2,8,13,2,3,-1.,8,14,2,1,3.,-1.6970280557870865e-003,0.5908237099647522,0.4500182867050171,0,2,14,12,2,3,-1.,14,13,2,1,3.,2.4118260480463505e-003,0.5315160751342773,0.3599720895290375,0,2,7,2,2,3,-1.,7,3,2,1,3.,-5.5300937965512276e-003,0.2334040999412537,0.4996814131736755,0,3,5,6,10,4,-1.,10,6,5,2,2.,5,8,5,2,2.,-2.6478730142116547e-003,0.5880935788154602,0.4684734046459198,0,2,9,13,1,6,-1.,9,16,1,3,2.,0.0112956296652555,0.4983777105808258,0.1884590983390808,0,3,10,12,2,2,-1.,11,12,1,1,2.,10,13,1,1,2.,-6.6952878842130303e-004,0.5872138142585754,0.4799019992351532,0,2,4,12,2,3,-1.,4,13,2,1,3.,1.4410680159926414e-003,0.5131189227104187,0.3501011133193970,0,2,14,4,6,6,-1.,14,6,6,2,3.,2.4637870956212282e-003,0.5339372158050537,0.4117639064788818,0,2,8,17,2,3,-1.,8,18,2,1,3.,3.3114518737420440e-004,0.4313383102416992,0.5398246049880981,0,2,16,4,4,6,-1.,16,6,4,2,3.,-0.0335572697222233,0.2675336897373200,0.5179154872894287,0,2,0,4,4,6,-1.,0,6,4,2,3.,0.0185394193977118,0.4973869919776917,0.2317177057266235,0,2,14,6,2,3,-1.,14,6,1,3,2.,-2.9698139405809343e-004,0.5529708266258240,0.4643664062023163,0,2,4,9,8,1,-1.,8,9,4,1,2.,-4.5577259152196348e-004,0.5629584193229675,0.4469191133975983,0,2,8,12,4,3,-1.,8,13,4,1,3.,-0.0101589802652597,0.6706212759017944,0.4925918877124786,0,2,5,12,10,6,-1.,5,14,10,2,3.,-2.2413829356082715e-005,0.5239421725273132,0.3912901878356934,0,2,11,12,1,2,-1.,11,13,1,1,2.,7.2034963523037732e-005,0.4799438118934631,0.5501788854598999,0,2,8,15,4,2,-1.,8,16,4,1,2.,-6.9267209619283676e-003,0.6930009722709656,0.4698084890842438,0,3,6,9,8,8,-1.,10,9,4,4,2.,6,13,4,4,2.,-7.6997838914394379e-003,0.4099623858928680,0.5480883121490479,0,3,7,12,4,6,-1.,7,12,2,3,2.,9,15,2,3,2.,-7.3130549862980843e-003,0.3283475935459137,0.5057886242866516,0,2,10,11,3,1,-1.,11,11,1,1,3.,1.9650589674711227e-003,0.4978047013282776,0.6398249864578247,0,3,9,7,2,10,-1.,9,7,1,5,2.,10,12,1,5,2.,7.1647600270807743e-003,0.4661160111427307,0.6222137212753296,0,2,8,0,6,6,-1.,10,0,2,6,3.,-0.0240786392241716,0.2334644943475723,0.5222162008285523,0,2,3,11,2,6,-1.,3,13,2,2,3.,-0.0210279691964388,0.1183653995394707,0.4938226044178009,0,2,16,12,1,2,-1.,16,13,1,1,2.,3.6017020465806127e-004,0.5325019955635071,0.4116711020469666,0,3,1,14,6,6,-1.,1,14,3,3,2.,4,17,3,3,2.,-0.0172197297215462,0.6278762221336365,0.4664269089698792,0,2,13,1,3,6,-1.,14,1,1,6,3.,-7.8672142699360847e-003,0.3403415083885193,0.5249736905097961,0,2,8,8,2,2,-1.,8,9,2,1,2.,-4.4777389848604798e-004,0.3610411882400513,0.5086259245872498,0,2,9,9,3,3,-1.,10,9,1,3,3.,5.5486010387539864e-003,0.4884265959262848,0.6203498244285584,0,2,8,7,3,3,-1.,8,8,3,1,3.,-6.9461148232221603e-003,0.2625930011272430,0.5011097192764282,0,2,14,0,2,3,-1.,14,0,1,3,2.,1.3569870498031378e-004,0.4340794980525971,0.5628312230110169,0,2,1,0,18,9,-1.,7,0,6,9,3.,-0.0458802506327629,0.6507998704910278,0.4696274995803833,0,2,11,5,4,15,-1.,11,5,2,15,2.,-0.0215825606137514,0.3826502859592438,0.5287616848945618,0,2,5,5,4,15,-1.,7,5,2,15,2.,-0.0202095396816731,0.3233368098735809,0.5074477195739746,0,2,14,0,2,3,-1.,14,0,1,3,2.,5.8496710844337940e-003,0.5177603960037231,0.4489670991897583,0,2,4,0,2,3,-1.,5,0,1,3,2.,-5.7476379879517481e-005,0.4020850956439972,0.5246363878250122,0,3,11,12,2,2,-1.,12,12,1,1,2.,11,13,1,1,2.,-1.1513100471347570e-003,0.6315072178840637,0.4905154109001160,0,3,7,12,2,2,-1.,7,12,1,1,2.,8,13,1,1,2.,1.9862831104546785e-003,0.4702459871768951,0.6497151255607605,0,2,12,0,3,4,-1.,13,0,1,4,3.,-5.2719512023031712e-003,0.3650383949279785,0.5227652788162231,0,2,4,11,3,3,-1.,4,12,3,1,3.,1.2662699446082115e-003,0.5166100859642029,0.3877618014812470,0,2,12,7,4,2,-1.,12,8,4,1,2.,-6.2919440679252148e-003,0.7375894188880920,0.5023847818374634,0,2,8,10,3,2,-1.,9,10,1,2,3.,6.7360111279413104e-004,0.4423226118087769,0.5495585799217224,0,2,9,9,3,2,-1.,10,9,1,2,3.,-1.0523450328037143e-003,0.5976396203041077,0.4859583079814911,0,2,8,9,3,2,-1.,9,9,1,2,3.,-4.4216238893568516e-004,0.5955939292907715,0.4398930966854096,0,2,12,0,3,4,-1.,13,0,1,4,3.,1.1747940443456173e-003,0.5349888205528259,0.4605058133602142,0,2,5,0,3,4,-1.,6,0,1,4,3.,5.2457437850534916e-003,0.5049191117286682,0.2941577136516571,0,3,4,14,12,4,-1.,10,14,6,2,2.,4,16,6,2,2.,-0.0245397202670574,0.2550177872180939,0.5218586921691895,0,2,8,13,2,3,-1.,8,14,2,1,3.,7.3793041519820690e-004,0.4424861073493958,0.5490816235542297,0,2,10,10,3,8,-1.,10,14,3,4,2.,1.4233799884095788e-003,0.5319514274597168,0.4081355929374695,0,3,8,10,4,8,-1.,8,10,2,4,2.,10,14,2,4,2.,-2.4149110540747643e-003,0.4087659120559692,0.5238950252532959,0,2,10,8,3,1,-1.,11,8,1,1,3.,-1.2165299849584699e-003,0.5674579143524170,0.4908052980899811,0,2,9,12,1,6,-1.,9,15,1,3,2.,-1.2438809499144554e-003,0.4129425883293152,0.5256118178367615,0,2,10,8,3,1,-1.,11,8,1,1,3.,6.1942739412188530e-003,0.5060194134712219,0.7313653230667114,0,2,7,8,3,1,-1.,8,8,1,1,3.,-1.6607169527560472e-003,0.5979632139205933,0.4596369862556458,0,2,5,2,15,14,-1.,5,9,15,7,2.,-0.0273162592202425,0.4174365103244782,0.5308842062950134,0,3,2,1,2,10,-1.,2,1,1,5,2.,3,6,1,5,2.,-1.5845570014789701e-003,0.5615804791450501,0.4519486129283905,0,2,14,14,2,3,-1.,14,15,2,1,3.,-1.5514739789068699e-003,0.4076187014579773,0.5360785126686096,0,2,2,7,3,3,-1.,3,7,1,3,3.,3.8446558755822480e-004,0.4347293972969055,0.5430442094802856,0,2,17,4,3,3,-1.,17,5,3,1,3.,-0.0146722598001361,0.1659304946660996,0.5146093964576721,0,2,0,4,3,3,-1.,0,5,3,1,3.,8.1608882173895836e-003,0.4961819052696228,0.1884745955467224,0,3,13,5,6,2,-1.,16,5,3,1,2.,13,6,3,1,2.,1.1121659772470593e-003,0.4868263900279999,0.6093816161155701,0,2,4,19,12,1,-1.,8,19,4,1,3.,-7.2603770531713963e-003,0.6284325122833252,0.4690375924110413,0,2,12,12,2,4,-1.,12,14,2,2,2.,-2.4046430189628154e-004,0.5575000047683716,0.4046044051647186,0,2,3,15,1,3,-1.,3,16,1,1,3.,-2.3348190006799996e-004,0.4115762114524841,0.5252848267555237,0,2,11,16,6,4,-1.,11,16,3,4,2.,5.5736480280756950e-003,0.4730072915554047,0.5690100789070129,0,2,2,10,3,10,-1.,3,10,1,10,3.,0.0306237693876028,0.4971886873245239,0.1740095019340515,0,2,12,8,2,4,-1.,12,8,1,4,2.,9.2074798885732889e-004,0.5372117757797241,0.4354872107505798,0,2,6,8,2,4,-1.,7,8,1,4,2.,-4.3550739064812660e-005,0.5366883873939514,0.4347316920757294,0,2,10,14,2,3,-1.,10,14,1,3,2.,-6.6452710889279842e-003,0.3435518145561218,0.5160533189773560,0,2,5,1,10,3,-1.,10,1,5,3,2.,0.0432219989597797,0.4766792058944702,0.7293652892112732,0,2,10,7,3,2,-1.,11,7,1,2,3.,2.2331769578158855e-003,0.5029315948486328,0.5633171200752258,0,2,5,6,9,2,-1.,8,6,3,2,3.,3.1829739455133677e-003,0.4016092121601105,0.5192136764526367,0,2,9,8,2,2,-1.,9,9,2,1,2.,-1.8027749320026487e-004,0.4088315963745117,0.5417919754981995,0,3,2,11,16,6,-1.,2,11,8,3,2.,10,14,8,3,2.,-5.2934689447283745e-003,0.4075677096843720,0.5243561863899231,0,3,12,7,2,2,-1.,13,7,1,1,2.,12,8,1,1,2.,1.2750959722325206e-003,0.4913282990455627,0.6387010812759399,0,2,9,5,2,3,-1.,9,6,2,1,3.,4.3385322205722332e-003,0.5031672120094299,0.2947346866130829,0,2,9,7,3,2,-1.,10,7,1,2,3.,8.5250744596123695e-003,0.4949789047241211,0.6308869123458862,0,2,5,1,8,12,-1.,5,7,8,6,2.,-9.4266352243721485e-004,0.5328366756439209,0.4285649955272675,0,2,13,5,2,2,-1.,13,6,2,1,2.,1.3609660090878606e-003,0.4991525113582611,0.5941501259803772,0,2,5,5,2,2,-1.,5,6,2,1,2.,4.4782509212382138e-004,0.4573504030704498,0.5854480862617493,0,2,12,4,3,3,-1.,12,5,3,1,3.,1.3360050506889820e-003,0.4604358971118927,0.5849052071571350,0,2,4,14,2,3,-1.,4,15,2,1,3.,-6.0967548051849008e-004,0.3969388902187347,0.5229423046112061,0,2,12,4,3,3,-1.,12,5,3,1,3.,-2.3656780831515789e-003,0.5808320045471191,0.4898357093334198,0,2,5,4,3,3,-1.,5,5,3,1,3.,1.0734340175986290e-003,0.4351210892200470,0.5470039248466492,0,3,9,14,2,6,-1.,10,14,1,3,2.,9,17,1,3,2.,2.1923359017819166e-003,0.5355060100555420,0.3842903971672058,0,2,8,14,3,2,-1.,9,14,1,2,3.,5.4968618787825108e-003,0.5018138885498047,0.2827191948890686,0,2,9,5,6,6,-1.,11,5,2,6,3.,-0.0753688216209412,0.1225076019763947,0.5148826837539673,0,2,5,5,6,6,-1.,7,5,2,6,3.,0.0251344703137875,0.4731766879558563,0.7025446295738220,0,2,13,13,1,2,-1.,13,14,1,1,2.,-2.9358599931583740e-005,0.5430532097816467,0.4656086862087250,0,2,0,2,10,2,-1.,0,3,10,1,2.,-5.8355910005047917e-004,0.4031040072441101,0.5190119743347168,0,2,13,13,1,2,-1.,13,14,1,1,2.,-2.6639450807124376e-003,0.4308126866817474,0.5161771178245544,0,3,5,7,2,2,-1.,5,7,1,1,2.,6,8,1,1,2.,-1.3804089976474643e-003,0.6219829916954041,0.4695515930652618,0,2,13,5,2,7,-1.,13,5,1,7,2.,1.2313219485804439e-003,0.5379363894462585,0.4425831139087677,0,2,6,13,1,2,-1.,6,14,1,1,2.,-1.4644179827882908e-005,0.5281640291213989,0.4222503006458283,0,2,11,0,3,7,-1.,12,0,1,7,3.,-0.0128188095986843,0.2582092881202698,0.5179932713508606,0,3,0,3,2,16,-1.,0,3,1,8,2.,1,11,1,8,2.,0.0228521898388863,0.4778693020343781,0.7609264254570007,0,2,11,0,3,7,-1.,12,0,1,7,3.,8.2305970136076212e-004,0.5340992212295532,0.4671724140644074,0,2,6,0,3,7,-1.,7,0,1,7,3.,0.0127701200544834,0.4965761005878449,0.1472366005182266,0,2,11,16,8,4,-1.,11,16,4,4,2.,-0.0500515103340149,0.6414994001388550,0.5016592144966126,0,2,1,16,8,4,-1.,5,16,4,4,2.,0.0157752707600594,0.4522320032119751,0.5685362219810486,0,2,13,5,2,7,-1.,13,5,1,7,2.,-0.0185016207396984,0.2764748930931091,0.5137959122657776,0,2,5,5,2,7,-1.,6,5,1,7,2.,2.4626250378787518e-003,0.5141941905021668,0.3795408010482788,0,2,18,6,2,14,-1.,18,13,2,7,2.,0.0629161670804024,0.5060648918151856,0.6580433845520020,0,2,6,10,3,4,-1.,6,12,3,2,2.,-2.1648500478477217e-005,0.5195388197898865,0.4019886851310730,0,2,14,7,1,2,-1.,14,8,1,1,2.,2.1180990152060986e-003,0.4962365031242371,0.5954458713531494,0,3,0,1,18,6,-1.,0,1,9,3,2.,9,4,9,3,2.,-0.0166348908096552,0.3757933080196381,0.5175446867942810,0,2,14,7,1,2,-1.,14,8,1,1,2.,-2.8899470344185829e-003,0.6624013781547546,0.5057178735733032,0,2,0,6,2,14,-1.,0,13,2,7,2.,0.0767832621932030,0.4795796871185303,0.8047714829444885,0,2,17,0,3,12,-1.,18,0,1,12,3.,3.9170677773654461e-003,0.4937882125377655,0.5719941854476929,0,2,0,6,18,3,-1.,0,7,18,1,3.,-0.0726706013083458,0.0538945607841015,0.4943903982639313,0,2,6,0,14,16,-1.,6,8,14,8,2.,0.5403950214385986,0.5129774212837219,0.1143338978290558,0,2,0,0,3,12,-1.,1,0,1,12,3.,2.9510019812732935e-003,0.4528343975543976,0.5698574185371399,0,2,13,0,3,7,-1.,14,0,1,7,3.,3.4508369863033295e-003,0.5357726812362671,0.4218730926513672,0,2,5,7,1,2,-1.,5,8,1,1,2.,-4.2077939724549651e-004,0.5916172862052918,0.4637925922870636,0,2,14,4,6,6,-1.,14,6,6,2,3.,3.3051050268113613e-003,0.5273385047912598,0.4382042884826660,0,2,5,7,7,2,-1.,5,8,7,1,2.,4.7735060798004270e-004,0.4046528041362763,0.5181884765625000,0,2,8,6,6,9,-1.,8,9,6,3,3.,-0.0259285103529692,0.7452235817909241,0.5089386105537415,0,2,5,4,6,1,-1.,7,4,2,1,3.,-2.9729790985584259e-003,0.3295435905456543,0.5058795213699341,0,3,13,0,6,4,-1.,16,0,3,2,2.,13,2,3,2,2.,5.8508329093456268e-003,0.4857144057750702,0.5793024897575378,0,2,1,2,18,12,-1.,1,6,18,4,3.,-0.0459675192832947,0.4312731027603149,0.5380653142929077,0,2,3,2,17,12,-1.,3,6,17,4,3.,0.1558596044778824,0.5196170210838318,0.1684713959693909,0,2,5,14,7,3,-1.,5,15,7,1,3.,0.0151648297905922,0.4735757112503052,0.6735026836395264,0,2,10,14,1,3,-1.,10,15,1,1,3.,-1.0604249546304345e-003,0.5822926759719849,0.4775702953338623,0,2,3,14,3,3,-1.,3,15,3,1,3.,6.6476291976869106e-003,0.4999198913574219,0.2319535017013550,0,2,14,4,6,6,-1.,14,6,6,2,3.,-0.0122311301529408,0.4750893115997315,0.5262982249259949,0,2,0,4,6,6,-1.,0,6,6,2,3.,5.6528882123529911e-003,0.5069767832756043,0.3561818897724152,0,2,12,5,4,3,-1.,12,6,4,1,3.,1.2977829901501536e-003,0.4875693917274475,0.5619062781333923,0,2,4,5,4,3,-1.,4,6,4,1,3.,0.0107815898954868,0.4750770032405853,0.6782308220863342,0,2,18,0,2,6,-1.,18,2,2,2,3.,2.8654779307544231e-003,0.5305461883544922,0.4290736019611359,0,2,8,1,4,9,-1.,10,1,2,9,2.,2.8663428965955973e-003,0.4518479108810425,0.5539351105690002,0,2,6,6,8,2,-1.,6,6,4,2,2.,-5.1983320154249668e-003,0.4149119853973389,0.5434188842773438,0,3,6,5,4,2,-1.,6,5,2,1,2.,8,6,2,1,2.,5.3739990107715130e-003,0.4717896878719330,0.6507657170295715,0,2,10,5,2,3,-1.,10,6,2,1,3.,-0.0146415298804641,0.2172164022922516,0.5161777138710022,0,2,9,5,1,3,-1.,9,6,1,1,3.,-1.5042580344015732e-005,0.5337383747100830,0.4298836886882782,0,2,9,10,2,2,-1.,9,11,2,1,2.,-1.1875660129589960e-004,0.4604594111442566,0.5582447052001953,0,2,0,8,4,3,-1.,0,9,4,1,3.,0.0169955305755138,0.4945895075798035,0.0738800764083862,0,2,6,0,8,6,-1.,6,3,8,3,2.,-0.0350959412753582,0.7005509138107300,0.4977591037750244,0,3,1,0,6,4,-1.,1,0,3,2,2.,4,2,3,2,2.,2.4217350874096155e-003,0.4466265141963959,0.5477694272994995,0,2,13,0,3,7,-1.,14,0,1,7,3.,-9.6340337768197060e-004,0.4714098870754242,0.5313338041305542,0,2,9,16,2,2,-1.,9,17,2,1,2.,1.6391130338888615e-004,0.4331546127796173,0.5342242121696472,0,2,11,4,6,10,-1.,11,9,6,5,2.,-0.0211414601653814,0.2644700109958649,0.5204498767852783,0,2,0,10,19,2,-1.,0,11,19,1,2.,8.7775202700868249e-004,0.5208349823951721,0.4152742922306061,0,2,9,5,8,9,-1.,9,8,8,3,3.,-0.0279439203441143,0.6344125270843506,0.5018811821937561,0,2,4,0,3,7,-1.,5,0,1,7,3.,6.7297378554940224e-003,0.5050438046455383,0.3500863909721375,0,3,8,6,4,12,-1.,10,6,2,6,2.,8,12,2,6,2.,0.0232810396701097,0.4966318011283875,0.6968677043914795,0,2,0,2,6,4,-1.,0,4,6,2,2.,-0.0116449799388647,0.3300260007381439,0.5049629807472229,0,2,8,15,4,3,-1.,8,16,4,1,3.,0.0157643090933561,0.4991598129272461,0.7321153879165649,0,2,8,0,3,7,-1.,9,0,1,7,3.,-1.3611479662358761e-003,0.3911735117435455,0.5160670876502991,0,2,9,5,3,4,-1.,10,5,1,4,3.,-8.1522337859496474e-004,0.5628911256790161,0.4949719011783600,0,2,8,5,3,4,-1.,9,5,1,4,3.,-6.0066272271797061e-004,0.5853595137596130,0.4550595879554749,0,2,7,6,6,1,-1.,9,6,2,1,3.,4.9715518252924085e-004,0.4271470010280609,0.5443599224090576,0,3,7,14,4,4,-1.,7,14,2,2,2.,9,16,2,2,2.,2.3475370835512877e-003,0.5143110752105713,0.3887656927108765,0,3,13,14,4,6,-1.,15,14,2,3,2.,13,17,2,3,2.,-8.9261569082736969e-003,0.6044502258300781,0.4971720874309540,0,2,7,8,1,8,-1.,7,12,1,4,2.,-0.0139199104160070,0.2583160996437073,0.5000367760658264,0,3,16,0,2,8,-1.,17,0,1,4,2.,16,4,1,4,2.,1.0209949687123299e-003,0.4857374131679535,0.5560358166694641,0,3,2,0,2,8,-1.,2,0,1,4,2.,3,4,1,4,2.,-2.7441629208624363e-003,0.5936884880065918,0.4645777046680450,0,2,6,1,14,3,-1.,6,2,14,1,3.,-0.0162001308053732,0.3163014948368073,0.5193495154380798,0,2,7,9,3,10,-1.,7,14,3,5,2.,4.3331980705261230e-003,0.5061224102973938,0.3458878993988037,0,2,9,14,2,2,-1.,9,15,2,1,2.,5.8497930876910686e-004,0.4779017865657806,0.5870177745819092,0,2,7,7,6,8,-1.,7,11,6,4,2.,-2.2466450463980436e-003,0.4297851026058197,0.5374773144721985,0,2,9,7,3,6,-1.,9,10,3,3,2.,2.3146099410951138e-003,0.5438671708106995,0.4640969932079315,0,2,7,13,3,3,-1.,7,14,3,1,3.,8.7679121643304825e-003,0.4726893007755280,0.6771789789199829,0,2,9,9,2,2,-1.,9,10,2,1,2.,-2.2448020172305405e-004,0.4229173064231873,0.5428048968315125,0,2,0,1,18,2,-1.,6,1,6,2,3.,-7.4336021207273006e-003,0.6098880767822266,0.4683673977851868,0,2,7,1,6,14,-1.,7,8,6,7,2.,-2.3189240600913763e-003,0.5689436793327332,0.4424242079257965,0,2,1,9,18,1,-1.,7,9,6,1,3.,-2.1042178850620985e-003,0.3762221038341522,0.5187087059020996,0,2,9,7,2,2,-1.,9,7,1,2,2.,4.6034841216169298e-004,0.4699405133724213,0.5771207213401794,0,2,9,3,2,9,-1.,10,3,1,9,2.,1.0547629790380597e-003,0.4465216994285584,0.5601701736450195,0,2,18,14,2,3,-1.,18,15,2,1,3.,8.7148818420246243e-004,0.5449805259704590,0.3914709091186523,0,2,7,11,3,1,-1.,8,11,1,1,3.,3.3364820410497487e-004,0.4564009010791779,0.5645738840103149,0,2,10,8,3,4,-1.,11,8,1,4,3.,-1.4853250468149781e-003,0.5747377872467041,0.4692778885364533,0,2,7,14,3,6,-1.,8,14,1,6,3.,3.0251620337367058e-003,0.5166196823120117,0.3762814104557037,0,2,10,8,3,4,-1.,11,8,1,4,3.,5.0280741415917873e-003,0.5002111792564392,0.6151527166366577,0,2,7,8,3,4,-1.,8,8,1,4,3.,-5.8164511574432254e-004,0.5394598245620728,0.4390751123428345,0,2,7,9,6,9,-1.,7,12,6,3,3.,0.0451415292918682,0.5188326835632324,0.2063035964965820,0,2,0,14,2,3,-1.,0,15,2,1,3.,-1.0795620037242770e-003,0.3904685080051422,0.5137907266616821,0,2,11,12,1,2,-1.,11,13,1,1,2.,1.5995999274309725e-004,0.4895322918891907,0.5427504181861877,0,2,4,3,8,3,-1.,8,3,4,3,2.,-0.0193592701107264,0.6975228786468506,0.4773507118225098,0,2,0,4,20,6,-1.,0,4,10,6,2.,0.2072550952434540,0.5233635902404785,0.3034991919994354,0,2,9,14,1,3,-1.,9,15,1,1,3.,-4.1953290929086506e-004,0.5419396758079529,0.4460186064243317,0,2,8,14,4,3,-1.,8,15,4,1,3.,2.2582069505006075e-003,0.4815764129161835,0.6027408838272095,0,2,0,15,14,4,-1.,0,17,14,2,2.,-6.7811207845807076e-003,0.3980278968811035,0.5183305740356445,0,2,1,14,18,6,-1.,1,17,18,3,2.,0.0111543098464608,0.5431231856346130,0.4188759922981262,0,3,0,0,10,6,-1.,0,0,5,3,2.,5,3,5,3,2.,0.0431624315679073,0.4738228023052216,0.6522961258888245];
    module.frontalface_alt = new Float32Array(classifier);
    module.frontalface_alt.tilted = true;
})(objectdetect);
/**
 * tracking.js - A modern approach for Computer Vision on the web.
 * @author Eduardo Lundgren <edu@rdo.io>
 * @version v1.0.0
 * @link http://trackingjs.com
 * @license BSD
 */
(function(window, undefined) {
  window.tracking = window.tracking || {};

  /**
   * Inherit the prototype methods from one constructor into another.
   *
   * Usage:
   * <pre>
   * function ParentClass(a, b) { }
   * ParentClass.prototype.foo = function(a) { }
   *
   * function ChildClass(a, b, c) {
   *   tracking.base(this, a, b);
   * }
   * tracking.inherits(ChildClass, ParentClass);
   *
   * var child = new ChildClass('a', 'b', 'c');
   * child.foo();
   * </pre>
   *
   * @param {Function} childCtor Child class.
   * @param {Function} parentCtor Parent class.
   */
  tracking.inherits = function(childCtor, parentCtor) {
    function TempCtor() {
    }
    TempCtor.prototype = parentCtor.prototype;
    childCtor.superClass_ = parentCtor.prototype;
    childCtor.prototype = new TempCtor();
    childCtor.prototype.constructor = childCtor;

    /**
     * Calls superclass constructor/method.
     *
     * This function is only available if you use tracking.inherits to express
     * inheritance relationships between classes.
     *
     * @param {!object} me Should always be "this".
     * @param {string} methodName The method name to call. Calling superclass
     *     constructor can be done with the special string 'constructor'.
     * @param {...*} var_args The arguments to pass to superclass
     *     method/constructor.
     * @return {*} The return value of the superclass method/constructor.
     */
    childCtor.base = function(me, methodName) {
      var args = Array.prototype.slice.call(arguments, 2);
      return parentCtor.prototype[methodName].apply(me, args);
    };
  };

  /**
   * Captures the user camera when tracking a video element and set its source
   * to the camera stream.
   * @param {HTMLVideoElement} element Canvas element to track.
   * @param {object} opt_options Optional configuration to the tracker.
   */
  tracking.initUserMedia_ = function(element, opt_options) {
    window.navigator.getUserMedia({
      video: true,
      audio: opt_options.audio
    }, function(stream) {
        try {
          element.src = window.URL.createObjectURL(stream);
        } catch (err) {
          element.src = stream;
        }
      }, function() {
        throw Error('Cannot capture user camera.');
      }
    );
  };

  /**
   * Tests whether the object is a dom node.
   * @param {object} o Object to be tested.
   * @return {boolean} True if the object is a dom node.
   */
  tracking.isNode = function(o) {
    return o.nodeType || this.isWindow(o);
  };

  /**
   * Tests whether the object is the `window` object.
   * @param {object} o Object to be tested.
   * @return {boolean} True if the object is the `window` object.
   */
  tracking.isWindow = function(o) {
    return !!(o && o.alert && o.document);
  };

  /**
   * Selects a dom node from a CSS3 selector using `document.querySelector`.
   * @param {string} selector
   * @param {object} opt_element The root element for the query. When not
   *     specified `document` is used as root element.
   * @return {HTMLElement} The first dom element that matches to the selector.
   *     If not found, returns `null`.
   */
  tracking.one = function(selector, opt_element) {
    if (this.isNode(selector)) {
      return selector;
    }
    return (opt_element || document).querySelector(selector);
  };

  /**
   * Tracks a canvas, image or video element based on the specified `tracker`
   * instance. This method extract the pixel information of the input element
   * to pass to the `tracker` instance. When tracking a video, the
   * `tracker.track(pixels, width, height)` will be in a
   * `requestAnimationFrame` loop in order to track all video frames.
   *
   * Example:
   * var tracker = new tracking.ColorTracker();
   *
   * tracking.track('#video', tracker);
   * or
   * tracking.track('#video', tracker, { camera: true });
   *
   * tracker.on('track', function(event) {
   *   // console.log(event.data[0].x, event.data[0].y)
   * });
   *
   * @param {HTMLElement} element The element to track, canvas, image or
   *     video.
   * @param {tracking.Tracker} tracker The tracker instance used to track the
   *     element.
   * @param {object} opt_options Optional configuration to the tracker.
   */
  tracking.track = function(element, tracker, opt_options) {
    element = tracking.one(element);
    if (!element) {
      throw new Error('Element not found, try a different element or selector.');
    }
    if (!tracker) {
      throw new Error('Tracker not specified, try `tracking.track(element, new tracking.FaceTracker())`.');
    }

    switch (element.nodeName.toLowerCase()) {
      case 'canvas':
        return this.trackCanvas_(element, tracker, opt_options);
      case 'img':
        return this.trackImg_(element, tracker, opt_options);
      case 'video':
        if (opt_options) {
          if (opt_options.camera) {
            this.initUserMedia_(element, opt_options);
          }
        }
        return this.trackVideo_(element, tracker, opt_options);
      default:
        throw new Error('Element not supported, try in a canvas, img, or video.');
    }
  };

  /**
   * Tracks a canvas element based on the specified `tracker` instance and
   * returns a `TrackerTask` for this track.
   * @param {HTMLCanvasElement} element Canvas element to track.
   * @param {tracking.Tracker} tracker The tracker instance used to track the
   *     element.
   * @param {object} opt_options Optional configuration to the tracker.
   * @return {tracking.TrackerTask}
   * @private
   */
  tracking.trackCanvas_ = function(element, tracker) {
    var self = this;
    var task = new tracking.TrackerTask(tracker);
    task.on('run', function() {
      self.trackCanvasInternal_(element, tracker);
    });
    return task.run();
  };

  /**
   * Tracks a canvas element based on the specified `tracker` instance. This
   * method extract the pixel information of the input element to pass to the
   * `tracker` instance.
   * @param {HTMLCanvasElement} element Canvas element to track.
   * @param {tracking.Tracker} tracker The tracker instance used to track the
   *     element.
   * @param {object} opt_options Optional configuration to the tracker.
   * @private
   */
  tracking.trackCanvasInternal_ = function(element, tracker) {
    var width = element.width;
    var height = element.height;
    var context = element.getContext('2d');
    var imageData = context.getImageData(0, 0, width, height);
    tracker.track(imageData.data, width, height);
  };

  /**
   * Tracks a image element based on the specified `tracker` instance. This
   * method extract the pixel information of the input element to pass to the
   * `tracker` instance.
   * @param {HTMLImageElement} element Canvas element to track.
   * @param {tracking.Tracker} tracker The tracker instance used to track the
   *     element.
   * @param {object} opt_options Optional configuration to the tracker.
   * @private
   */
  tracking.trackImg_ = function(element, tracker) {
    var width = element.width;
    var height = element.height;
    var canvas = document.createElement('canvas');

    canvas.width = width;
    canvas.height = height;

    var task = new tracking.TrackerTask(tracker);
    task.on('run', function() {
      tracking.Canvas.loadImage(canvas, element.src, 0, 0, width, height, function() {
        tracking.trackCanvasInternal_(canvas, tracker);
      });
    });
    return task.run();
  };

  /**
   * Tracks a video element based on the specified `tracker` instance. This
   * method extract the pixel information of the input element to pass to the
   * `tracker` instance. The `tracker.track(pixels, width, height)` will be in
   * a `requestAnimationFrame` loop in order to track all video frames.
   * @param {HTMLVideoElement} element Canvas element to track.
   * @param {tracking.Tracker} tracker The tracker instance used to track the
   *     element.
   * @param {object} opt_options Optional configuration to the tracker.
   * @private
   */
  tracking.trackVideo_ = function(element, tracker) {
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    var width;
    var height;

    var resizeCanvas_ = function() {
      width = element.offsetWidth;
      height = element.offsetHeight;
      canvas.width = width;
      canvas.height = height;
    };
    resizeCanvas_();
    element.addEventListener('resize', resizeCanvas_);

    var requestId;
    var requestAnimationFrame_ = function() {
      requestId = window.requestAnimationFrame(function() {
        if (element.readyState === element.HAVE_ENOUGH_DATA) {
          try {
            // Firefox v~30.0 gets confused with the video readyState firing an
            // erroneous HAVE_ENOUGH_DATA just before HAVE_CURRENT_DATA state,
            // hence keep trying to read it until resolved.
            context.drawImage(element, 0, 0, width, height);
          } catch (err) {}
          tracking.trackCanvasInternal_(canvas, tracker);
        }
        requestAnimationFrame_();
      });
    };

    var task = new tracking.TrackerTask(tracker);
    task.on('stop', function() {
      window.cancelAnimationFrame(requestId);
    });
    task.on('run', function() {
      requestAnimationFrame_();
    });
    return task.run();
  };

  // Browser polyfills
  //===================

  if (!window.URL) {
    window.URL = window.URL || window.webkitURL || window.msURL || window.oURL;
  }

  if (!navigator.getUserMedia) {
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia || navigator.msGetUserMedia;
  }
}(window));

(function() {
  /**
   * EventEmitter utility.
   * @constructor
   */
  tracking.EventEmitter = function() {};

  /**
   * Holds event listeners scoped by event type.
   * @type {object}
   * @private
   */
  tracking.EventEmitter.prototype.events_ = null;

  /**
   * Adds a listener to the end of the listeners array for the specified event.
   * @param {string} event
   * @param {function} listener
   * @return {object} Returns emitter, so calls can be chained.
   */
  tracking.EventEmitter.prototype.addListener = function(event, listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('Listener must be a function');
    }
    if (!this.events_) {
      this.events_ = {};
    }

    this.emit('newListener', event, listener);

    if (!this.events_[event]) {
      this.events_[event] = [];
    }

    this.events_[event].push(listener);

    return this;
  };

  /**
   * Returns an array of listeners for the specified event.
   * @param {string} event
   * @return {array} Array of listeners.
   */
  tracking.EventEmitter.prototype.listeners = function(event) {
    return this.events_ && this.events_[event];
  };

  /**
   * Execute each of the listeners in order with the supplied arguments.
   * @param {string} event
   * @param {*} opt_args [arg1], [arg2], [...]
   * @return {boolean} Returns true if event had listeners, false otherwise.
   */
  tracking.EventEmitter.prototype.emit = function(event) {
    var listeners = this.listeners(event);
    if (listeners) {
      var args = Array.prototype.slice.call(arguments, 1);
      for (var i = 0; i < listeners.length; i++) {
        if (listeners[i]) {
          listeners[i].apply(this, args);
        }
      }
      return true;
    }
    return false;
  };

  /**
   * Adds a listener to the end of the listeners array for the specified event.
   * @param {string} event
   * @param {function} listener
   * @return {object} Returns emitter, so calls can be chained.
   */
  tracking.EventEmitter.prototype.on = tracking.EventEmitter.prototype.addListener;

  /**
   * Adds a one time listener for the event. This listener is invoked only the
   * next time the event is fired, after which it is removed.
   * @param {string} event
   * @param {function} listener
   * @return {object} Returns emitter, so calls can be chained.
   */
  tracking.EventEmitter.prototype.once = function(event, listener) {
    var self = this;
    self.on(event, function handlerInternal() {
      self.removeListener(event, handlerInternal);
      listener.apply(this, arguments);
    });
  };

  /**
   * Removes all listeners, or those of the specified event. It's not a good
   * idea to remove listeners that were added elsewhere in the code,
   * especially when it's on an emitter that you didn't create.
   * @param {string} event
   * @return {object} Returns emitter, so calls can be chained.
   */
  tracking.EventEmitter.prototype.removeAllListeners = function(opt_event) {
    if (!this.events_) {
      return this;
    }
    if (opt_event) {
      delete this.events_[opt_event];
    } else {
      delete this.events_;
    }
    return this;
  };

  /**
   * Remove a listener from the listener array for the specified event.
   * Caution: changes array indices in the listener array behind the listener.
   * @param {string} event
   * @param {function} listener
   * @return {object} Returns emitter, so calls can be chained.
   */
  tracking.EventEmitter.prototype.removeListener = function(event, listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('Listener must be a function');
    }
    if (!this.events_) {
      return this;
    }

    var listeners = this.listeners(event);
    if (Array.isArray(listeners)) {
      var i = listeners.indexOf(listener);
      if (i < 0) {
        return this;
      }
      listeners.splice(i, 1);
    }

    return this;
  };

  /**
   * By default EventEmitters will print a warning if more than 10 listeners
   * are added for a particular event. This is a useful default which helps
   * finding memory leaks. Obviously not all Emitters should be limited to 10.
   * This function allows that to be increased. Set to zero for unlimited.
   * @param {number} n The maximum number of listeners.
   */
  tracking.EventEmitter.prototype.setMaxListeners = function() {
    throw new Error('Not implemented');
  };

}());

(function() {
  /**
   * Canvas utility.
   * @static
   * @constructor
   */
  tracking.Canvas = {};

  /**
   * Loads an image source into the canvas.
   * @param {HTMLCanvasElement} canvas The canvas dom element.
   * @param {string} src The image source.
   * @param {number} x The canvas horizontal coordinate to load the image.
   * @param {number} y The canvas vertical coordinate to load the image.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {function} opt_callback Callback that fires when the image is loaded
   *     into the canvas.
   * @static
   */
  tracking.Canvas.loadImage = function(canvas, src, x, y, width, height, opt_callback) {
    var instance = this;
    var img = new window.Image();

    img.onload = function() {
      var context = canvas.getContext('2d');
      canvas.width = width;
      canvas.height = height;
      context.drawImage(img, x, y, width, height);
      if (opt_callback) {
        opt_callback.call(instance);
      }
      img = null;
    };
    img.src = src;
  };
}());

(function() {
  /**
   * DisjointSet utility with path compression. Some applications involve
   * grouping n distinct objects into a collection of disjoint sets. Two
   * important operations are then finding which set a given object belongs to
   * and uniting the two sets. A disjoint set data structure maintains a
   * collection S={ S1 , S2 ,..., Sk } of disjoint dynamic sets. Each set is
   * identified by a representative, which usually is a member in the set.
   * @static
   * @constructor
   */
  tracking.DisjointSet = function(length) {
    if (length === undefined) {
      throw new Error('DisjointSet length not specified.');
    }
    this.length = length;
    this.parent = new Uint32Array(length);
    for (var i = 0; i < length; i++) {
      this.parent[i] = i;
    }
  };

  /**
   * Holds the length of the internal set.
   * @type {number}
   */
  tracking.DisjointSet.prototype.length = null;

  /**
   * Holds the set containing the representative values.
   * @type {Array.<number>}
   */
  tracking.DisjointSet.prototype.parent = null;

  /**
   * Finds a pointer to the representative of the set containing i.
   * @param {number} i
   * @return {number} The representative set of i.
   */
  tracking.DisjointSet.prototype.find = function(i) {
    if (this.parent[i] === i) {
      return i;
    } else {
      return (this.parent[i] = this.find(this.parent[i]));
    }
  };

  /**
   * Unites two dynamic sets containing objects i and j, say Si and Sj, into
   * a new set that Si âˆª Sj, assuming that Si âˆ© Sj = âˆ…;
   * @param {number} i
   * @param {number} j
   */
  tracking.DisjointSet.prototype.union = function(i, j) {
    var iRepresentative = this.find(i);
    var jRepresentative = this.find(j);
    this.parent[iRepresentative] = jRepresentative;
  };

}());

(function() {
  /**
   * Image utility.
   * @static
   * @constructor
   */
  tracking.Image = {};

  /**
   * Computes gaussian blur. Adpated from
   * https://github.com/kig/canvasfilters.
   * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {number} diameter Gaussian blur diameter, must be greater than 1.
   * @return {array} The edge pixels in a linear [r,g,b,a,...] array.
   */
  tracking.Image.blur = function(pixels, width, height, diameter) {
    diameter = Math.abs(diameter);
    if (diameter <= 1) {
      throw new Error('Diameter should be greater than 1.');
    }
    var radius = diameter / 2;
    var len = Math.ceil(diameter) + (1 - (Math.ceil(diameter) % 2));
    var weights = new Float32Array(len);
    var rho = (radius + 0.5) / 3;
    var rhoSq = rho * rho;
    var gaussianFactor = 1 / Math.sqrt(2 * Math.PI * rhoSq);
    var rhoFactor = -1 / (2 * rho * rho);
    var wsum = 0;
    var middle = Math.floor(len / 2);
    for (var i = 0; i < len; i++) {
      var x = i - middle;
      var gx = gaussianFactor * Math.exp(x * x * rhoFactor);
      weights[i] = gx;
      wsum += gx;
    }
    for (var j = 0; j < weights.length; j++) {
      weights[j] /= wsum;
    }
    return this.separableConvolve(pixels, width, height, weights, weights, false);
  };

  /**
   * Computes the integral image for summed, squared, rotated and sobel pixels.
   * @param {array} pixels The pixels in a linear [r,g,b,a,...] array to loop
   *     through.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {array} opt_integralImage Empty array of size `width * height` to
   *     be filled with the integral image values. If not specified compute sum
   *     values will be skipped.
   * @param {array} opt_integralImageSquare Empty array of size `width *
   *     height` to be filled with the integral image squared values. If not
   *     specified compute squared values will be skipped.
   * @param {array} opt_tiltedIntegralImage Empty array of size `width *
   *     height` to be filled with the rotated integral image values. If not
   *     specified compute sum values will be skipped.
   * @param {array} opt_integralImageSobel Empty array of size `width *
   *     height` to be filled with the integral image of sobel values. If not
   *     specified compute sobel filtering will be skipped.
   * @static
   */
  tracking.Image.computeIntegralImage = function(pixels, width, height, opt_integralImage, opt_integralImageSquare, opt_tiltedIntegralImage, opt_integralImageSobel) {
    if (arguments.length < 4) {
      throw new Error('You should specify at least one output array in the order: sum, square, tilted, sobel.');
    }
    var pixelsSobel;
    if (opt_integralImageSobel) {
      pixelsSobel = tracking.Image.sobel(pixels, width, height);
    }
    for (var i = 0; i < height; i++) {
      for (var j = 0; j < width; j++) {
        var w = i * width * 4 + j * 4;
        var pixel = ~~(pixels[w] * 0.299 + pixels[w + 1] * 0.587 + pixels[w + 2] * 0.114);
        if (opt_integralImage) {
          this.computePixelValueSAT_(opt_integralImage, width, i, j, pixel);
        }
        if (opt_integralImageSquare) {
          this.computePixelValueSAT_(opt_integralImageSquare, width, i, j, pixel * pixel);
        }
        if (opt_tiltedIntegralImage) {
          var w1 = w - width * 4;
          var pixelAbove = ~~(pixels[w1] * 0.299 + pixels[w1 + 1] * 0.587 + pixels[w1 + 2] * 0.114);
          this.computePixelValueRSAT_(opt_tiltedIntegralImage, width, i, j, pixel, pixelAbove || 0);
        }
        if (opt_integralImageSobel) {
          this.computePixelValueSAT_(opt_integralImageSobel, width, i, j, pixelsSobel[w]);
        }
      }
    }
  };

  /**
   * Helper method to compute the rotated summed area table (RSAT) by the
   * formula:
   *
   * RSAT(x, y) = RSAT(x-1, y-1) + RSAT(x+1, y-1) - RSAT(x, y-2) + I(x, y) + I(x, y-1)
   *
   * @param {number} width The image width.
   * @param {array} RSAT Empty array of size `width * height` to be filled with
   *     the integral image values. If not specified compute sum values will be
   *     skipped.
   * @param {number} i Vertical position of the pixel to be evaluated.
   * @param {number} j Horizontal position of the pixel to be evaluated.
   * @param {number} pixel Pixel value to be added to the integral image.
   * @static
   * @private
   */
  tracking.Image.computePixelValueRSAT_ = function(RSAT, width, i, j, pixel, pixelAbove) {
    var w = i * width + j;
    RSAT[w] = (RSAT[w - width - 1] || 0) + (RSAT[w - width + 1] || 0) - (RSAT[w - width - width] || 0) + pixel + pixelAbove;
  };

  /**
   * Helper method to compute the summed area table (SAT) by the formula:
   *
   * SAT(x, y) = SAT(x, y-1) + SAT(x-1, y) + I(x, y) - SAT(x-1, y-1)
   *
   * @param {number} width The image width.
   * @param {array} SAT Empty array of size `width * height` to be filled with
   *     the integral image values. If not specified compute sum values will be
   *     skipped.
   * @param {number} i Vertical position of the pixel to be evaluated.
   * @param {number} j Horizontal position of the pixel to be evaluated.
   * @param {number} pixel Pixel value to be added to the integral image.
   * @static
   * @private
   */
  tracking.Image.computePixelValueSAT_ = function(SAT, width, i, j, pixel) {
    var w = i * width + j;
    SAT[w] = (SAT[w - width] || 0) + (SAT[w - 1] || 0) + pixel - (SAT[w - width - 1] || 0);
  };

  /**
   * Converts a color from a colorspace based on an RGB color model to a
   * grayscale representation of its luminance. The coefficients represent the
   * measured intensity perception of typical trichromat humans, in
   * particular, human vision is most sensitive to green and least sensitive
   * to blue.
   * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {boolean} fillRGBA If the result should fill all RGBA values with the gray scale
   *  values, instead of returning a single value per pixel.
   * @param {Uint8ClampedArray} The grayscale pixels in a linear array ([p,p,p,a,...] if fillRGBA
   *  is true and [p1, p2, p3, ...] if fillRGBA is false).
   * @static
   */
  tracking.Image.grayscale = function(pixels, width, height, fillRGBA) {
    var gray = new Uint8ClampedArray(fillRGBA ? pixels.length : pixels.length >> 2);
    var p = 0;
    var w = 0;
    for (var i = 0; i < height; i++) {
      for (var j = 0; j < width; j++) {
        var value = pixels[w] * 0.299 + pixels[w + 1] * 0.587 + pixels[w + 2] * 0.114;
        gray[p++] = value;

        if (fillRGBA) {
          gray[p++] = value;
          gray[p++] = value;
          gray[p++] = pixels[w + 3];
        }

        w += 4;
      }
    }
    return gray;
  };

  /**
   * Fast horizontal separable convolution. A point spread function (PSF) is
   * said to be separable if it can be broken into two one-dimensional
   * signals: a vertical and a horizontal projection. The convolution is
   * performed by sliding the kernel over the image, generally starting at the
   * top left corner, so as to move the kernel through all the positions where
   * the kernel fits entirely within the boundaries of the image. Adpated from
   * https://github.com/kig/canvasfilters.
   * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {array} weightsVector The weighting vector, e.g [-1,0,1].
   * @param {number} opaque
   * @return {array} The convoluted pixels in a linear [r,g,b,a,...] array.
   */
  tracking.Image.horizontalConvolve = function(pixels, width, height, weightsVector, opaque) {
    var side = weightsVector.length;
    var halfSide = Math.floor(side / 2);
    var output = new Float32Array(width * height * 4);
    var alphaFac = opaque ? 1 : 0;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        var sy = y;
        var sx = x;
        var offset = (y * width + x) * 4;
        var r = 0;
        var g = 0;
        var b = 0;
        var a = 0;
        for (var cx = 0; cx < side; cx++) {
          var scy = sy;
          var scx = Math.min(width - 1, Math.max(0, sx + cx - halfSide));
          var poffset = (scy * width + scx) * 4;
          var wt = weightsVector[cx];
          r += pixels[poffset] * wt;
          g += pixels[poffset + 1] * wt;
          b += pixels[poffset + 2] * wt;
          a += pixels[poffset + 3] * wt;
        }
        output[offset] = r;
        output[offset + 1] = g;
        output[offset + 2] = b;
        output[offset + 3] = a + alphaFac * (255 - a);
      }
    }
    return output;
  };

  /**
   * Fast vertical separable convolution. A point spread function (PSF) is
   * said to be separable if it can be broken into two one-dimensional
   * signals: a vertical and a horizontal projection. The convolution is
   * performed by sliding the kernel over the image, generally starting at the
   * top left corner, so as to move the kernel through all the positions where
   * the kernel fits entirely within the boundaries of the image. Adpated from
   * https://github.com/kig/canvasfilters.
   * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {array} weightsVector The weighting vector, e.g [-1,0,1].
   * @param {number} opaque
   * @return {array} The convoluted pixels in a linear [r,g,b,a,...] array.
   */
  tracking.Image.verticalConvolve = function(pixels, width, height, weightsVector, opaque) {
    var side = weightsVector.length;
    var halfSide = Math.floor(side / 2);
    var output = new Float32Array(width * height * 4);
    var alphaFac = opaque ? 1 : 0;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        var sy = y;
        var sx = x;
        var offset = (y * width + x) * 4;
        var r = 0;
        var g = 0;
        var b = 0;
        var a = 0;
        for (var cy = 0; cy < side; cy++) {
          var scy = Math.min(height - 1, Math.max(0, sy + cy - halfSide));
          var scx = sx;
          var poffset = (scy * width + scx) * 4;
          var wt = weightsVector[cy];
          r += pixels[poffset] * wt;
          g += pixels[poffset + 1] * wt;
          b += pixels[poffset + 2] * wt;
          a += pixels[poffset + 3] * wt;
        }
        output[offset] = r;
        output[offset + 1] = g;
        output[offset + 2] = b;
        output[offset + 3] = a + alphaFac * (255 - a);
      }
    }
    return output;
  };

  /**
   * Fast separable convolution. A point spread function (PSF) is said to be
   * separable if it can be broken into two one-dimensional signals: a
   * vertical and a horizontal projection. The convolution is performed by
   * sliding the kernel over the image, generally starting at the top left
   * corner, so as to move the kernel through all the positions where the
   * kernel fits entirely within the boundaries of the image. Adpated from
   * https://github.com/kig/canvasfilters.
   * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {array} horizWeights The horizontal weighting vector, e.g [-1,0,1].
   * @param {array} vertWeights The vertical vector, e.g [-1,0,1].
   * @param {number} opaque
   * @return {array} The convoluted pixels in a linear [r,g,b,a,...] array.
   */
  tracking.Image.separableConvolve = function(pixels, width, height, horizWeights, vertWeights, opaque) {
    var vertical = this.verticalConvolve(pixels, width, height, vertWeights, opaque);
    return this.horizontalConvolve(vertical, width, height, horizWeights, opaque);
  };

  /**
   * Compute image edges using Sobel operator. Computes the vertical and
   * horizontal gradients of the image and combines the computed images to
   * find edges in the image. The way we implement the Sobel filter here is by
   * first grayscaling the image, then taking the horizontal and vertical
   * gradients and finally combining the gradient images to make up the final
   * image. Adpated from https://github.com/kig/canvasfilters.
   * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @return {array} The edge pixels in a linear [r,g,b,a,...] array.
   */
  tracking.Image.sobel = function(pixels, width, height) {
    pixels = this.grayscale(pixels, width, height, true);
    var output = new Float32Array(width * height * 4);
    var sobelSignVector = new Float32Array([-1, 0, 1]);
    var sobelScaleVector = new Float32Array([1, 2, 1]);
    var vertical = this.separableConvolve(pixels, width, height, sobelSignVector, sobelScaleVector);
    var horizontal = this.separableConvolve(pixels, width, height, sobelScaleVector, sobelSignVector);

    for (var i = 0; i < output.length; i += 4) {
      var v = vertical[i];
      var h = horizontal[i];
      var p = Math.sqrt(h * h + v * v);
      output[i] = p;
      output[i + 1] = p;
      output[i + 2] = p;
      output[i + 3] = 255;
    }

    return output;
  };

}());

(function() {
  /**
   * ViolaJones utility.
   * @static
   * @constructor
   */
  tracking.ViolaJones = {};

  /**
   * Holds the minimum area of intersection that defines when a rectangle is
   * from the same group. Often when a face is matched multiple rectangles are
   * classified as possible rectangles to represent the face, when they
   * intersects they are grouped as one face.
   * @type {number}
   * @default 0.5
   * @static
   */
  tracking.ViolaJones.REGIONS_OVERLAP = 0.5;

  /**
   * Holds the HAAR cascade classifiers converted from OpenCV training.
   * @type {array}
   * @static
   */
  tracking.ViolaJones.classifiers = {};

  /**
   * Detects through the HAAR cascade data rectangles matches.
   * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {number} initialScale The initial scale to start the block
   *     scaling.
   * @param {number} scaleFactor The scale factor to scale the feature block.
   * @param {number} stepSize The block step size.
   * @param {number} edgesDensity Percentage density edges inside the
   *     classifier block. Value from [0.0, 1.0], defaults to 0.2. If specified
   *     edge detection will be applied to the image to prune dead areas of the
   *     image, this can improve significantly performance.
   * @param {number} data The HAAR cascade data.
   * @return {array} Found rectangles.
   * @static
   */
  tracking.ViolaJones.detect = function(pixels, width, height, initialScale, scaleFactor, stepSize, edgesDensity, data) {
    var total = 0;
    var rects = [];
    var integralImage = new Int32Array(width * height);
    var integralImageSquare = new Int32Array(width * height);
    var tiltedIntegralImage = new Int32Array(width * height);

    var integralImageSobel;
    if (edgesDensity > 0) {
      integralImageSobel = new Int32Array(width * height);
    }

    tracking.Image.computeIntegralImage(pixels, width, height, integralImage, integralImageSquare, tiltedIntegralImage, integralImageSobel);

    var minWidth = data[0];
    var minHeight = data[1];
    var scale = initialScale * scaleFactor;
    var blockWidth = (scale * minWidth) | 0;
    var blockHeight = (scale * minHeight) | 0;

    while (blockWidth < width && blockHeight < height) {
      var step = (scale * stepSize + 0.5) | 0;
      for (var i = 0; i < (height - blockHeight); i += step) {
        for (var j = 0; j < (width - blockWidth); j += step) {

          if (edgesDensity > 0) {
            if (this.isTriviallyExcluded(edgesDensity, integralImageSobel, i, j, width, blockWidth, blockHeight)) {
              continue;
            }
          }

          if (this.evalStages_(data, integralImage, integralImageSquare, tiltedIntegralImage, i, j, width, blockWidth, blockHeight, scale)) {
            rects[total++] = {
              width: blockWidth,
              height: blockHeight,
              x: j,
              y: i
            };
          }
        }
      }

      scale *= scaleFactor;
      blockWidth = (scale * minWidth) | 0;
      blockHeight = (scale * minHeight) | 0;
    }
    return this.mergeRectangles_(rects);
  };

  /**
   * Fast check to test whether the edges density inside the block is greater
   * than a threshold, if true it tests the stages. This can improve
   * significantly performance.
   * @param {number} edgesDensity Percentage density edges inside the
   *     classifier block.
   * @param {array} integralImageSobel The integral image of a sobel image.
   * @param {number} i Vertical position of the pixel to be evaluated.
   * @param {number} j Horizontal position of the pixel to be evaluated.
   * @param {number} width The image width.
   * @return {boolean} True whether the block at position i,j can be skipped,
   *     false otherwise.
   * @static
   * @protected
   */
  tracking.ViolaJones.isTriviallyExcluded = function(edgesDensity, integralImageSobel, i, j, width, blockWidth, blockHeight) {
    var wbA = i * width + j;
    var wbB = wbA + blockWidth;
    var wbD = wbA + blockHeight * width;
    var wbC = wbD + blockWidth;
    var blockEdgesDensity = (integralImageSobel[wbA] - integralImageSobel[wbB] - integralImageSobel[wbD] + integralImageSobel[wbC]) / (blockWidth * blockHeight * 255);
    if (blockEdgesDensity < edgesDensity) {
      return true;
    }
    return false;
  };

  /**
   * Evaluates if the block size on i,j position is a valid HAAR cascade
   * stage.
   * @param {number} data The HAAR cascade data.
   * @param {number} i Vertical position of the pixel to be evaluated.
   * @param {number} j Horizontal position of the pixel to be evaluated.
   * @param {number} width The image width.
   * @param {number} blockSize The block size.
   * @param {number} scale The scale factor of the block size and its original
   *     size.
   * @param {number} inverseArea The inverse area of the block size.
   * @return {boolean} Whether the region passes all the stage tests.
   * @private
   * @static
   */
  tracking.ViolaJones.evalStages_ = function(data, integralImage, integralImageSquare, tiltedIntegralImage, i, j, width, blockWidth, blockHeight, scale) {
    var inverseArea = 1.0 / (blockWidth * blockHeight);
    var wbA = i * width + j;
    var wbB = wbA + blockWidth;
    var wbD = wbA + blockHeight * width;
    var wbC = wbD + blockWidth;
    var mean = (integralImage[wbA] - integralImage[wbB] - integralImage[wbD] + integralImage[wbC]) * inverseArea;
    var variance = (integralImageSquare[wbA] - integralImageSquare[wbB] - integralImageSquare[wbD] + integralImageSquare[wbC]) * inverseArea - mean * mean;

    var standardDeviation = 1;
    if (variance > 0) {
      standardDeviation = Math.sqrt(variance);
    }

    var length = data.length;

    for (var w = 2; w < length; ) {
      var stageSum = 0;
      var stageThreshold = data[w++];
      var nodeLength = data[w++];

      while (nodeLength--) {
        var rectsSum = 0;
        var tilted = data[w++];
        var rectsLength = data[w++];

        for (var r = 0; r < rectsLength; r++) {
          var rectLeft = (j + data[w++] * scale + 0.5) | 0;
          var rectTop = (i + data[w++] * scale + 0.5) | 0;
          var rectWidth = (data[w++] * scale + 0.5) | 0;
          var rectHeight = (data[w++] * scale + 0.5) | 0;
          var rectWeight = data[w++];

          var w1;
          var w2;
          var w3;
          var w4;
          if (tilted) {
            // RectSum(r) = RSAT(x-h+w, y+w+h-1) + RSAT(x, y-1) - RSAT(x-h, y+h-1) - RSAT(x+w, y+w-1)
            w1 = (rectLeft - rectHeight + rectWidth) + (rectTop + rectWidth + rectHeight - 1) * width;
            w2 = rectLeft + (rectTop - 1) * width;
            w3 = (rectLeft - rectHeight) + (rectTop + rectHeight - 1) * width;
            w4 = (rectLeft + rectWidth) + (rectTop + rectWidth - 1) * width;
            rectsSum += (tiltedIntegralImage[w1] + tiltedIntegralImage[w2] - tiltedIntegralImage[w3] - tiltedIntegralImage[w4]) * rectWeight;
          } else {
            // RectSum(r) = SAT(x-1, y-1) + SAT(x+w-1, y+h-1) - SAT(x-1, y+h-1) - SAT(x+w-1, y-1)
            w1 = rectTop * width + rectLeft;
            w2 = w1 + rectWidth;
            w3 = w1 + rectHeight * width;
            w4 = w3 + rectWidth;
            rectsSum += (integralImage[w1] - integralImage[w2] - integralImage[w3] + integralImage[w4]) * rectWeight;
            // TODO: Review the code below to analyze performance when using it instead.
            // w1 = (rectLeft - 1) + (rectTop - 1) * width;
            // w2 = (rectLeft + rectWidth - 1) + (rectTop + rectHeight - 1) * width;
            // w3 = (rectLeft - 1) + (rectTop + rectHeight - 1) * width;
            // w4 = (rectLeft + rectWidth - 1) + (rectTop - 1) * width;
            // rectsSum += (integralImage[w1] + integralImage[w2] - integralImage[w3] - integralImage[w4]) * rectWeight;
          }
        }

        var nodeThreshold = data[w++];
        var nodeLeft = data[w++];
        var nodeRight = data[w++];

        if (rectsSum * inverseArea < nodeThreshold * standardDeviation) {
          stageSum += nodeLeft;
        } else {
          stageSum += nodeRight;
        }
      }

      if (stageSum < stageThreshold) {
        return false;
      }
    }
    return true;
  };

  /**
   * Postprocess the detected sub-windows in order to combine overlapping
   * detections into a single detection.
   * @param {array} rects
   * @return {array}
   * @private
   * @static
   */
  tracking.ViolaJones.mergeRectangles_ = function(rects) {
    var disjointSet = new tracking.DisjointSet(rects.length);

    for (var i = 0; i < rects.length; i++) {
      var r1 = rects[i];
      for (var j = 0; j < rects.length; j++) {
        var r2 = rects[j];
        if (tracking.Math.intersectRect(r1.x, r1.y, r1.x + r1.width, r1.y + r1.height, r2.x, r2.y, r2.x + r2.width, r2.y + r2.height)) {
          var x1 = Math.max(r1.x, r2.x);
          var y1 = Math.max(r1.y, r2.y);
          var x2 = Math.min(r1.x + r1.width, r2.x + r2.width);
          var y2 = Math.min(r1.y + r1.height, r2.y + r2.height);
          var overlap = (x1 - x2) * (y1 - y2);
          var area1 = (r1.width * r1.height);
          var area2 = (r2.width * r2.height);

          if ((overlap / (area1 * (area1 / area2)) >= this.REGIONS_OVERLAP) &&
            (overlap / (area2 * (area1 / area2)) >= this.REGIONS_OVERLAP)) {
            disjointSet.union(i, j);
          }
        }
      }
    }

    var map = {};
    for (var k = 0; k < disjointSet.length; k++) {
      var rep = disjointSet.find(k);
      if (!map[rep]) {
        map[rep] = {
          total: 1,
          width: rects[k].width,
          height: rects[k].height,
          x: rects[k].x,
          y: rects[k].y
        };
        continue;
      }
      map[rep].total++;
      map[rep].width += rects[k].width;
      map[rep].height += rects[k].height;
      map[rep].x += rects[k].x;
      map[rep].y += rects[k].y;
    }

    var result = [];
    Object.keys(map).forEach(function(key) {
      var rect = map[key];
      result.push({
        total: rect.total,
        width: (rect.width / rect.total + 0.5) | 0,
        height: (rect.height / rect.total + 0.5) | 0,
        x: (rect.x / rect.total + 0.5) | 0,
        y: (rect.y / rect.total + 0.5) | 0
      });
    });

    return result;
  };

}());

(function() {
  /**
   * Brief intends for "Binary Robust Independent Elementary Features".This
   * method generates a binary string for each keypoint found by an extractor
   * method.
   * @static
   * @constructor
   */
  tracking.Brief = {};

  /**
   * The set of binary tests is defined by the nd (x,y)-location pairs
   * uniquely chosen during the initialization. Values could vary between N =
   * 128,256,512. N=128 yield good compromises between speed, storage
   * efficiency, and recognition rate.
   * @type {number}
   */
  tracking.Brief.N = 512;

  /**
   * Caches coordinates values of (x,y)-location pairs uniquely chosen during
   * the initialization.
   * @type {Object.<number, Int32Array>}
   * @private
   * @static
   */
  tracking.Brief.randomImageOffsets_ = {};

  /**
   * Caches delta values of (x,y)-location pairs uniquely chosen during
   * the initialization.
   * @type {Int32Array}
   * @private
   * @static
   */
  tracking.Brief.randomWindowOffsets_ = null;

  /**
   * Generates a brinary string for each found keypoints extracted using an
   * extractor method.
   * @param {array} The grayscale pixels in a linear [p1,p2,...] array.
   * @param {number} width The image width.
   * @param {array} keypoints
   * @return {Int32Array} Returns an array where for each four sequence int
   *     values represent the descriptor binary string (128 bits) necessary
   *     to describe the corner, e.g. [0,0,0,0, 0,0,0,0, ...].
   * @static
   */
  tracking.Brief.getDescriptors = function(pixels, width, keypoints) {
    // Optimizing divide by 32 operation using binary shift
    // (this.N >> 5) === this.N/32.
    var descriptors = new Int32Array((keypoints.length >> 1) * (this.N >> 5));
    var descriptorWord = 0;
    var offsets = this.getRandomOffsets_(width);
    var position = 0;

    for (var i = 0; i < keypoints.length; i += 2) {
      var w = width * keypoints[i + 1] + keypoints[i];

      var offsetsPosition = 0;
      for (var j = 0, n = this.N; j < n; j++) {
        if (pixels[offsets[offsetsPosition++] + w] < pixels[offsets[offsetsPosition++] + w]) {
          // The bit in the position `j % 32` of descriptorWord should be set to 1. We do
          // this by making an OR operation with a binary number that only has the bit
          // in that position set to 1. That binary number is obtained by shifting 1 left by
          // `j % 32` (which is the same as `j & 31` left) positions.
          descriptorWord |= 1 << (j & 31);
        }

        // If the next j is a multiple of 32, we will need to use a new descriptor word to hold
        // the next results.
        if (!((j + 1) & 31)) {
          descriptors[position++] = descriptorWord;
          descriptorWord = 0;
        }
      }
    }

    return descriptors;
  };

  /**
   * Matches sets of features {mi} and {mâ€²j} extracted from two images taken
   * from similar, and often successive, viewpoints. A classical procedure
   * runs as follows. For each point {mi} in the first image, search in a
   * region of the second image around location {mi} for point {mâ€²j}. The
   * search is based on the similarity of the local image windows, also known
   * as kernel windows, centered on the points, which strongly characterizes
   * the points when the images are sufficiently close. Once each keypoint is
   * described with its binary string, they need to be compared with the
   * closest matching point. Distance metric is critical to the performance of
   * in- trusion detection systems. Thus using binary strings reduces the size
   * of the descriptor and provides an interesting data structure that is fast
   * to operate whose similarity can be measured by the Hamming distance.
   * @param {array} keypoints1
   * @param {array} descriptors1
   * @param {array} keypoints2
   * @param {array} descriptors2
   * @return {Int32Array} Returns an array where the index is the corner1
   *     index coordinate, and the value is the corresponding match index of
   *     corner2, e.g. keypoints1=[x0,y0,x1,y1,...] and
   *     keypoints2=[x'0,y'0,x'1,y'1,...], if x0 matches x'1 and x1 matches x'0,
   *     the return array would be [3,0].
   * @static
   */
  tracking.Brief.match = function(keypoints1, descriptors1, keypoints2, descriptors2) {
    var len1 = keypoints1.length >> 1;
    var len2 = keypoints2.length >> 1;
    var matches = new Array(len1);

    for (var i = 0; i < len1; i++) {
      var min = Infinity;
      var minj = 0;
      for (var j = 0; j < len2; j++) {
        var dist = 0;
        // Optimizing divide by 32 operation using binary shift
        // (this.N >> 5) === this.N/32.
        for (var k = 0, n = this.N >> 5; k < n; k++) {
          dist += tracking.Math.hammingWeight(descriptors1[i * n + k] ^ descriptors2[j * n + k]);
        }
        if (dist < min) {
          min = dist;
          minj = j;
        }
      }
      matches[i] = {
        index1: i,
        index2: minj,
        keypoint1: [keypoints1[2 * i], keypoints1[2 * i + 1]],
        keypoint2: [keypoints2[2 * minj], keypoints2[2 * minj + 1]],
        confidence: 1 - min / this.N
      };
    }

    return matches;
  };

  /**
   * Removes matches outliers by testing matches on both directions.
   * @param {array} keypoints1
   * @param {array} descriptors1
   * @param {array} keypoints2
   * @param {array} descriptors2
   * @return {Int32Array} Returns an array where the index is the corner1
   *     index coordinate, and the value is the corresponding match index of
   *     corner2, e.g. keypoints1=[x0,y0,x1,y1,...] and
   *     keypoints2=[x'0,y'0,x'1,y'1,...], if x0 matches x'1 and x1 matches x'0,
   *     the return array would be [3,0].
   * @static
   */
  tracking.Brief.reciprocalMatch = function(keypoints1, descriptors1, keypoints2, descriptors2) {
    var matches = [];
    if (keypoints1.length === 0 || keypoints2.length === 0) {
      return matches;
    }

    var matches1 = tracking.Brief.match(keypoints1, descriptors1, keypoints2, descriptors2);
    var matches2 = tracking.Brief.match(keypoints2, descriptors2, keypoints1, descriptors1);
    for (var i = 0; i < matches1.length; i++) {
      if (matches2[matches1[i].index2].index2 === i) {
        matches.push(matches1[i]);
      }
    }
    return matches;
  };

  /**
   * Gets the coordinates values of (x,y)-location pairs uniquely chosen
   * during the initialization.
   * @return {array} Array with the random offset values.
   * @private
   */
  tracking.Brief.getRandomOffsets_ = function(width) {
    if (!this.randomWindowOffsets_) {
      var windowPosition = 0;
      var windowOffsets = new Int32Array(4 * this.N);
      for (var i = 0; i < this.N; i++) {
        windowOffsets[windowPosition++] = Math.round(tracking.Math.uniformRandom(-15, 16));
        windowOffsets[windowPosition++] = Math.round(tracking.Math.uniformRandom(-15, 16));
        windowOffsets[windowPosition++] = Math.round(tracking.Math.uniformRandom(-15, 16));
        windowOffsets[windowPosition++] = Math.round(tracking.Math.uniformRandom(-15, 16));
      }
      this.randomWindowOffsets_ = windowOffsets;
    }

    if (!this.randomImageOffsets_[width]) {
      var imagePosition = 0;
      var imageOffsets = new Int32Array(2 * this.N);
      for (var j = 0; j < this.N; j++) {
        imageOffsets[imagePosition++] = this.randomWindowOffsets_[4 * j] * width + this.randomWindowOffsets_[4 * j + 1];
        imageOffsets[imagePosition++] = this.randomWindowOffsets_[4 * j + 2] * width + this.randomWindowOffsets_[4 * j + 3];
      }
      this.randomImageOffsets_[width] = imageOffsets;
    }

    return this.randomImageOffsets_[width];
  };
}());

(function() {
  /**
   * FAST intends for "Features from Accelerated Segment Test". This method
   * performs a point segment test corner detection. The segment test
   * criterion operates by considering a circle of sixteen pixels around the
   * corner candidate p. The detector classifies p as a corner if there exists
   * a set of n contiguous pixelsin the circle which are all brighter than the
   * intensity of the candidate pixel Ip plus a threshold t, or all darker
   * than Ip âˆ’ t.
   *
   *       15 00 01
   *    14          02
   * 13                03
   * 12       []       04
   * 11                05
   *    10          06
   *       09 08 07
   *
   * For more reference:
   * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.60.3991&rep=rep1&type=pdf
   * @static
   * @constructor
   */
  tracking.Fast = {};

  /**
   * Holds the threshold to determine whether the tested pixel is brighter or
   * darker than the corner candidate p.
   * @type {number}
   * @default 40
   * @static
   */
  tracking.Fast.THRESHOLD = 40;

  /**
   * Caches coordinates values of the circle surounding the pixel candidate p.
   * @type {Object.<number, Int32Array>}
   * @private
   * @static
   */
  tracking.Fast.circles_ = {};

  /**
   * Finds corners coordinates on the graysacaled image.
   * @param {array} The grayscale pixels in a linear [p1,p2,...] array.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {number} threshold to determine whether the tested pixel is brighter or
   *     darker than the corner candidate p. Default value is 40.
   * @return {array} Array containing the coordinates of all found corners,
   *     e.g. [x0,y0,x1,y1,...], where P(x0,y0) represents a corner coordinate.
   * @static
   */
  tracking.Fast.findCorners = function(pixels, width, height, opt_threshold) {
    var circleOffsets = this.getCircleOffsets_(width);
    var circlePixels = new Int32Array(16);
    var corners = [];

    if (opt_threshold === undefined) {
      opt_threshold = this.THRESHOLD;
    }

    // When looping through the image pixels, skips the first three lines from
    // the image boundaries to constrain the surrounding circle inside the image
    // area.
    for (var i = 3; i < height - 3; i++) {
      for (var j = 3; j < width - 3; j++) {
        var w = i * width + j;
        var p = pixels[w];

        // Loops the circle offsets to read the pixel value for the sixteen
        // surrounding pixels.
        for (var k = 0; k < 16; k++) {
          circlePixels[k] = pixels[w + circleOffsets[k]];
        }

        if (this.isCorner(p, circlePixels, opt_threshold)) {
          // The pixel p is classified as a corner, as optimization increment j
          // by the circle radius 3 to skip the neighbor pixels inside the
          // surrounding circle. This can be removed without compromising the
          // result.
          corners.push(j, i);
          j += 3;
        }
      }
    }

    return corners;
  };

  /**
   * Checks if the circle pixel is brigther than the candidate pixel p by
   * a threshold.
   * @param {number} circlePixel The circle pixel value.
   * @param {number} p The value of the candidate pixel p.
   * @param {number} threshold
   * @return {Boolean}
   * @static
   */
  tracking.Fast.isBrighter = function(circlePixel, p, threshold) {
    return circlePixel - p > threshold;
  };

  /**
   * Checks if the circle pixel is within the corner of the candidate pixel p
   * by a threshold.
   * @param {number} p The value of the candidate pixel p.
   * @param {number} circlePixel The circle pixel value.
   * @param {number} threshold
   * @return {Boolean}
   * @static
   */
  tracking.Fast.isCorner = function(p, circlePixels, threshold) {
    if (this.isTriviallyExcluded(circlePixels, p, threshold)) {
      return false;
    }

    for (var x = 0; x < 16; x++) {
      var darker = true;
      var brighter = true;

      for (var y = 0; y < 9; y++) {
        var circlePixel = circlePixels[(x + y) & 15];

        if (!this.isBrighter(p, circlePixel, threshold)) {
          brighter = false;
          if (darker === false) {
            break;
          }
        }

        if (!this.isDarker(p, circlePixel, threshold)) {
          darker = false;
          if (brighter === false) {
            break;
          }
        }
      }

      if (brighter || darker) {
        return true;
      }
    }

    return false;
  };

  /**
   * Checks if the circle pixel is darker than the candidate pixel p by
   * a threshold.
   * @param {number} circlePixel The circle pixel value.
   * @param {number} p The value of the candidate pixel p.
   * @param {number} threshold
   * @return {Boolean}
   * @static
   */
  tracking.Fast.isDarker = function(circlePixel, p, threshold) {
    return p - circlePixel > threshold;
  };

  /**
   * Fast check to test if the candidate pixel is a trivially excluded value.
   * In order to be a corner, the candidate pixel value should be darker or
   * brigther than 9-12 surrouding pixels, when at least three of the top,
   * bottom, left and right pixels are brither or darker it can be
   * automatically excluded improving the performance.
   * @param {number} circlePixel The circle pixel value.
   * @param {number} p The value of the candidate pixel p.
   * @param {number} threshold
   * @return {Boolean}
   * @static
   * @protected
   */
  tracking.Fast.isTriviallyExcluded = function(circlePixels, p, threshold) {
    var count = 0;
    var circleBottom = circlePixels[8];
    var circleLeft = circlePixels[12];
    var circleRight = circlePixels[4];
    var circleTop = circlePixels[0];

    if (this.isBrighter(circleTop, p, threshold)) {
      count++;
    }
    if (this.isBrighter(circleRight, p, threshold)) {
      count++;
    }
    if (this.isBrighter(circleBottom, p, threshold)) {
      count++;
    }
    if (this.isBrighter(circleLeft, p, threshold)) {
      count++;
    }

    if (count < 3) {
      count = 0;
      if (this.isDarker(circleTop, p, threshold)) {
        count++;
      }
      if (this.isDarker(circleRight, p, threshold)) {
        count++;
      }
      if (this.isDarker(circleBottom, p, threshold)) {
        count++;
      }
      if (this.isDarker(circleLeft, p, threshold)) {
        count++;
      }
      if (count < 3) {
        return true;
      }
    }

    return false;
  };

  /**
   * Gets the sixteen offset values of the circle surrounding pixel.
   * @param {number} width The image width.
   * @return {array} Array with the sixteen offset values of the circle
   *     surrounding pixel.
   * @private
   */
  tracking.Fast.getCircleOffsets_ = function(width) {
    if (this.circles_[width]) {
      return this.circles_[width];
    }

    var circle = new Int32Array(16);

    circle[0] = -width - width - width;
    circle[1] = circle[0] + 1;
    circle[2] = circle[1] + width + 1;
    circle[3] = circle[2] + width + 1;
    circle[4] = circle[3] + width;
    circle[5] = circle[4] + width;
    circle[6] = circle[5] + width - 1;
    circle[7] = circle[6] + width - 1;
    circle[8] = circle[7] - 1;
    circle[9] = circle[8] - 1;
    circle[10] = circle[9] - width - 1;
    circle[11] = circle[10] - width - 1;
    circle[12] = circle[11] - width;
    circle[13] = circle[12] - width;
    circle[14] = circle[13] - width + 1;
    circle[15] = circle[14] - width + 1;

    this.circles_[width] = circle;
    return circle;
  };
}());

(function() {
  /**
   * Math utility.
   * @static
   * @constructor
   */
  tracking.Math = {};

  /**
   * Euclidean distance between two points P(x0, y0) and P(x1, y1).
   * @param {number} x0 Horizontal coordinate of P0.
   * @param {number} y0 Vertical coordinate of P0.
   * @param {number} x1 Horizontal coordinate of P1.
   * @param {number} y1 Vertical coordinate of P1.
   * @return {number} The euclidean distance.
   */
  tracking.Math.distance = function(x0, y0, x1, y1) {
    var dx = x1 - x0;
    var dy = y1 - y0;

    return Math.sqrt(dx * dx + dy * dy);
  };

  /**
   * Calculates the Hamming weight of a string, which is the number of symbols that are
   * different from the zero-symbol of the alphabet used. It is thus
   * equivalent to the Hamming distance from the all-zero string of the same
   * length. For the most typical case, a string of bits, this is the number
   * of 1's in the string.
   *
   * Example:
   *
   * <pre>
   *  Binary string     Hamming weight
   *   11101                 4
   *   11101010              5
   * </pre>
   *
   * @param {number} i Number that holds the binary string to extract the hamming weight.
   * @return {number} The hamming weight.
   */
  tracking.Math.hammingWeight = function(i) {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);

    return ((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
  };

  /**
   * Generates a random number between [a, b] interval.
   * @param {number} a
   * @param {number} b
   * @return {number}
   */
  tracking.Math.uniformRandom = function(a, b) {
    return a + Math.random() * (b - a);
  };

  /**
   * Tests if a rectangle intersects with another.
   *
   *  <pre>
   *  x0y0 --------       x2y2 --------
   *      |       |           |       |
   *      -------- x1y1       -------- x3y3
   * </pre>
   *
   * @param {number} x0 Horizontal coordinate of P0.
   * @param {number} y0 Vertical coordinate of P0.
   * @param {number} x1 Horizontal coordinate of P1.
   * @param {number} y1 Vertical coordinate of P1.
   * @param {number} x2 Horizontal coordinate of P2.
   * @param {number} y2 Vertical coordinate of P2.
   * @param {number} x3 Horizontal coordinate of P3.
   * @param {number} y3 Vertical coordinate of P3.
   * @return {boolean}
   */
  tracking.Math.intersectRect = function(x0, y0, x1, y1, x2, y2, x3, y3) {
    return !(x2 > x1 || x3 < x0 || y2 > y1 || y3 < y0);
  };

}());

(function() {
  /**
   * Matrix utility.
   * @static
   * @constructor
   */
  tracking.Matrix = {};

  /**
   * Loops the array organized as major-row order and executes `fn` callback
   * for each iteration. The `fn` callback receives the following parameters:
   * `(r,g,b,a,index,i,j)`, where `r,g,b,a` represents the pixel color with
   * alpha channel, `index` represents the position in the major-row order
   * array and `i,j` the respective indexes positions in two dimentions.
   * @param {array} pixels The pixels in a linear [r,g,b,a,...] array to loop
   *     through.
   * @param {number} width The image width.
   * @param {number} height The image height.
   * @param {function} fn The callback function for each pixel.
   * @param {number} opt_jump Optional jump for the iteration, by default it
   *     is 1, hence loops all the pixels of the array.
   * @static
   */
  tracking.Matrix.forEach = function(pixels, width, height, fn, opt_jump) {
    opt_jump = opt_jump || 1;
    for (var i = 0; i < height; i += opt_jump) {
      for (var j = 0; j < width; j += opt_jump) {
        var w = i * width * 4 + j * 4;
        fn.call(this, pixels[w], pixels[w + 1], pixels[w + 2], pixels[w + 3], w, i, j);
      }
    }
  };

}());

(function() {
  /**
   * EPnp utility.
   * @static
   * @constructor
   */
  tracking.EPnP = {};

  tracking.EPnP.solve = function(objectPoints, imagePoints, cameraMatrix) {};
}());

(function() {
  /**
   * Tracker utility.
   * @constructor
   * @extends {tracking.EventEmitter}
   */
  tracking.Tracker = function() {
    tracking.Tracker.base(this, 'constructor');
  };

  tracking.inherits(tracking.Tracker, tracking.EventEmitter);

  /**
   * Tracks the pixels on the array. This method is called for each video
   * frame in order to emit `track` event.
   * @param {Uint8ClampedArray} pixels The pixels data to track.
   * @param {number} width The pixels canvas width.
   * @param {number} height The pixels canvas height.
   */
  tracking.Tracker.prototype.track = function() {};
}());

(function() {
  /**
   * TrackerTask utility.
   * @constructor
   * @extends {tracking.EventEmitter}
   */
  tracking.TrackerTask = function(tracker) {
    tracking.TrackerTask.base(this, 'constructor');

    if (!tracker) {
      throw new Error('Tracker instance not specified.');
    }

    this.setTracker(tracker);
  };

  tracking.inherits(tracking.TrackerTask, tracking.EventEmitter);

  /**
   * Holds the tracker instance managed by this task.
   * @type {tracking.Tracker}
   * @private
   */
  tracking.TrackerTask.prototype.tracker_ = null;

  /**
   * Holds if the tracker task is in running.
   * @type {boolean}
   * @private
   */
  tracking.TrackerTask.prototype.running_ = false;

  /**
   * Gets the tracker instance managed by this task.
   * @return {tracking.Tracker}
   */
  tracking.TrackerTask.prototype.getTracker = function() {
    return this.tracker_;
  };

  /**
   * Returns true if the tracker task is in running, false otherwise.
   * @return {boolean}
   * @private
   */
  tracking.TrackerTask.prototype.inRunning = function() {
    return this.running_;
  };

  /**
   * Sets if the tracker task is in running.
   * @param {boolean} running
   * @private
   */
  tracking.TrackerTask.prototype.setRunning = function(running) {
    this.running_ = running;
  };

  /**
   * Sets the tracker instance managed by this task.
   * @return {tracking.Tracker}
   */
  tracking.TrackerTask.prototype.setTracker = function(tracker) {
    this.tracker_ = tracker;
  };

  /**
   * Emits a `run` event on the tracker task for the implementers to run any
   * child action, e.g. `requestAnimationFrame`.
   * @return {object} Returns itself, so calls can be chained.
   */
  tracking.TrackerTask.prototype.run = function() {
    var self = this;

    if (this.inRunning()) {
      return;
    }

    this.setRunning(true);
    this.reemitTrackEvent_ = function(event) {
      self.emit('track', event);
    };
    this.tracker_.on('track', this.reemitTrackEvent_);
    this.emit('run');
    return this;
  };

  /**
   * Emits a `stop` event on the tracker task for the implementers to stop any
   * child action being done, e.g. `requestAnimationFrame`.
   * @return {object} Returns itself, so calls can be chained.
   */
  tracking.TrackerTask.prototype.stop = function() {
    if (!this.inRunning()) {
      return;
    }

    this.setRunning(false);
    this.emit('stop');
    this.tracker_.removeListener('track', this.reemitTrackEvent_);
    return this;
  };
}());

(function() {
  /**
   * ColorTracker utility to track colored blobs in a frrame using color
   * difference evaluation.
   * @constructor
   * @param {string|Array.<string>} opt_colors Optional colors to track.
   * @extends {tracking.Tracker}
   */
  tracking.ColorTracker = function(opt_colors) {
    tracking.ColorTracker.base(this, 'constructor');

    if (typeof opt_colors === 'string') {
      opt_colors = [opt_colors];
    }

    if (opt_colors) {
      opt_colors.forEach(function(color) {
        if (!tracking.ColorTracker.getColor(color)) {
          throw new Error('Color not valid, try `new tracking.ColorTracker("magenta")`.');
        }
      });
      this.setColors(opt_colors);
    }
  };

  tracking.inherits(tracking.ColorTracker, tracking.Tracker);

  /**
   * Holds the known colors.
   * @type {Object.<string, function>}
   * @private
   * @static
   */
  tracking.ColorTracker.knownColors_ = {};

  /**
   * Caches coordinates values of the neighbours surrounding a pixel.
   * @type {Object.<number, Int32Array>}
   * @private
   * @static
   */
  tracking.ColorTracker.neighbours_ = {};

  /**
   * Registers a color as known color.
   * @param {string} name The color name.
   * @param {function} fn The color function to test if the passed (r,g,b) is
   *     the desired color.
   * @static
   */
  tracking.ColorTracker.registerColor = function(name, fn) {
    tracking.ColorTracker.knownColors_[name] = fn;
  };

  /**
   * Gets the known color function that is able to test whether an (r,g,b) is
   * the desired color.
   * @param {string} name The color name.
   * @return {function} The known color test function.
   * @static
   */
  tracking.ColorTracker.getColor = function(name) {
    return tracking.ColorTracker.knownColors_[name];
  };

  /**
   * Holds the colors to be tracked by the `ColorTracker` instance.
   * @default ['magenta']
   * @type {Array.<string>}
   */
  tracking.ColorTracker.prototype.colors = ['magenta'];

  /**
   * Holds the minimum dimension to classify a rectangle.
   * @default 20
   * @type {number}
   */
  tracking.ColorTracker.prototype.minDimension = 20;

  /**
   * Holds the maximum dimension to classify a rectangle.
   * @default Infinity
   * @type {number}
   */
  tracking.ColorTracker.prototype.maxDimension = Infinity;


  /**
   * Holds the minimum group size to be classified as a rectangle.
   * @default 30
   * @type {number}
   */
  tracking.ColorTracker.prototype.minGroupSize = 30;

  /**
   * Calculates the central coordinate from the cloud points. The cloud points
   * are all points that matches the desired color.
   * @param {Array.<number>} cloud Major row order array containing all the
   *     points from the desired color, e.g. [x1, y1, c2, y2, ...].
   * @param {number} total Total numbers of pixels of the desired color.
   * @return {object} Object contaning the x, y and estimated z coordinate of
   *     the blog extracted from the cloud points.
   * @private
   */
  tracking.ColorTracker.prototype.calculateDimensions_ = function(cloud, total) {
    var maxx = -1;
    var maxy = -1;
    var minx = Infinity;
    var miny = Infinity;

    for (var c = 0; c < total; c += 2) {
      var x = cloud[c];
      var y = cloud[c + 1];

      if (x < minx) {
        minx = x;
      }
      if (x > maxx) {
        maxx = x;
      }
      if (y < miny) {
        miny = y;
      }
      if (y > maxy) {
        maxy = y;
      }
    }

    return {
      width: maxx - minx,
      height: maxy - miny,
      x: minx,
      y: miny
    };
  };

  /**
   * Gets the colors being tracked by the `ColorTracker` instance.
   * @return {Array.<string>}
   */
  tracking.ColorTracker.prototype.getColors = function() {
    return this.colors;
  };

  /**
   * Gets the minimum dimension to classify a rectangle.
   * @return {number}
   */
  tracking.ColorTracker.prototype.getMinDimension = function() {
    return this.minDimension;
  };

  /**
   * Gets the maximum dimension to classify a rectangle.
   * @return {number}
   */
  tracking.ColorTracker.prototype.getMaxDimension = function() {
    return this.maxDimension;
  };

  /**
   * Gets the minimum group size to be classified as a rectangle.
   * @return {number}
   */
  tracking.ColorTracker.prototype.getMinGroupSize = function() {
    return this.minGroupSize;
  };

  /**
   * Gets the eight offset values of the neighbours surrounding a pixel.
   * @param {number} width The image width.
   * @return {array} Array with the eight offset values of the neighbours
   *     surrounding a pixel.
   * @private
   */
  tracking.ColorTracker.prototype.getNeighboursForWidth_ = function(width) {
    if (tracking.ColorTracker.neighbours_[width]) {
      return tracking.ColorTracker.neighbours_[width];
    }

    var neighbours = new Int32Array(8);

    neighbours[0] = -width * 4;
    neighbours[1] = -width * 4 + 4;
    neighbours[2] = 4;
    neighbours[3] = width * 4 + 4;
    neighbours[4] = width * 4;
    neighbours[5] = width * 4 - 4;
    neighbours[6] = -4;
    neighbours[7] = -width * 4 - 4;

    tracking.ColorTracker.neighbours_[width] = neighbours;

    return neighbours;
  };

  /**
   * Unites groups whose bounding box intersect with each other.
   * @param {Array.<Object>} rects
   * @private
   */
  tracking.ColorTracker.prototype.mergeRectangles_ = function(rects) {
    var intersects;
    var results = [];
    var minDimension = this.getMinDimension();
    var maxDimension = this.getMaxDimension();

    for (var r = 0; r < rects.length; r++) {
      var r1 = rects[r];
      intersects = true;
      for (var s = r + 1; s < rects.length; s++) {
        var r2 = rects[s];
        if (tracking.Math.intersectRect(r1.x, r1.y, r1.x + r1.width, r1.y + r1.height, r2.x, r2.y, r2.x + r2.width, r2.y + r2.height)) {
          intersects = false;
          var x1 = Math.min(r1.x, r2.x);
          var y1 = Math.min(r1.y, r2.y);
          var x2 = Math.max(r1.x + r1.width, r2.x + r2.width);
          var y2 = Math.max(r1.y + r1.height, r2.y + r2.height);
          r2.height = y2 - y1;
          r2.width = x2 - x1;
          r2.x = x1;
          r2.y = y1;
          break;
        }
      }

      if (intersects) {
        if (r1.width >= minDimension && r1.height >= minDimension) {
          if (r1.width <= maxDimension && r1.height <= maxDimension) {
            results.push(r1);
          }
        }
      }
    }

    return results;
  };

  /**
   * Sets the colors to be tracked by the `ColorTracker` instance.
   * @param {Array.<string>} colors
   */
  tracking.ColorTracker.prototype.setColors = function(colors) {
    this.colors = colors;
  };

  /**
   * Sets the minimum dimension to classify a rectangle.
   * @param {number} minDimension
   */
  tracking.ColorTracker.prototype.setMinDimension = function(minDimension) {
    this.minDimension = minDimension;
  };

  /**
   * Sets the maximum dimension to classify a rectangle.
   * @param {number} maxDimension
   */
  tracking.ColorTracker.prototype.setMaxDimension = function(maxDimension) {
    this.maxDimension = maxDimension;
  };

  /**
   * Sets the minimum group size to be classified as a rectangle.
   * @param {number} minGroupSize
   */
  tracking.ColorTracker.prototype.setMinGroupSize = function(minGroupSize) {
    this.minGroupSize = minGroupSize;
  };

  /**
   * Tracks the `Video` frames. This method is called for each video frame in
   * order to emit `track` event.
   * @param {Uint8ClampedArray} pixels The pixels data to track.
   * @param {number} width The pixels canvas width.
   * @param {number} height The pixels canvas height.
   */
  tracking.ColorTracker.prototype.track = function(pixels, width, height) {
    var self = this;
    var colors = this.getColors();

    if (!colors) {
      throw new Error('Colors not specified, try `new tracking.ColorTracker("magenta")`.');
    }

    var results = [];

    colors.forEach(function(color) {
      results = results.concat(self.trackColor_(pixels, width, height, color));
    });

    this.emit('track', {
      data: results
    });
  };

  /**
   * Find the given color in the given matrix of pixels using Flood fill
   * algorithm to determines the area connected to a given node in a
   * multi-dimensional array.
   * @param {Uint8ClampedArray} pixels The pixels data to track.
   * @param {number} width The pixels canvas width.
   * @param {number} height The pixels canvas height.
   * @param {string} color The color to be found
   * @private
   */
  tracking.ColorTracker.prototype.trackColor_ = function(pixels, width, height, color) {
    var colorFn = tracking.ColorTracker.knownColors_[color];
    var currGroup = new Int32Array(pixels.length >> 2);
    var currGroupSize;
    var currI;
    var currJ;
    var currW;
    var marked = new Int8Array(pixels.length);
    var minGroupSize = this.getMinGroupSize();
    var neighboursW = this.getNeighboursForWidth_(width);
    var queue = new Int32Array(pixels.length);
    var queuePosition;
    var results = [];
    var w = -4;

    if (!colorFn) {
      return results;
    }

    for (var i = 0; i < height; i++) {
      for (var j = 0; j < width; j++) {
        w += 4;

        if (marked[w]) {
          continue;
        }

        currGroupSize = 0;

        queuePosition = -1;
        queue[++queuePosition] = w;
        queue[++queuePosition] = i;
        queue[++queuePosition] = j;

        marked[w] = 1;

        while (queuePosition >= 0) {
          currJ = queue[queuePosition--];
          currI = queue[queuePosition--];
          currW = queue[queuePosition--];

          if (colorFn(pixels[currW], pixels[currW + 1], pixels[currW + 2], pixels[currW + 3], currW, currI, currJ)) {
            currGroup[currGroupSize++] = currJ;
            currGroup[currGroupSize++] = currI;

            for (var k = 0; k < neighboursW.length; k++) {
              var otherW = currW + neighboursW[k];
              var otherI = currI + neighboursI[k];
              var otherJ = currJ + neighboursJ[k];
              if (!marked[otherW] && otherI >= 0 && otherI < height && otherJ >= 0 && otherJ < width) {
                queue[++queuePosition] = otherW;
                queue[++queuePosition] = otherI;
                queue[++queuePosition] = otherJ;

                marked[otherW] = 1;
              }
            }
          }
        }

        if (currGroupSize >= minGroupSize) {
          var data = this.calculateDimensions_(currGroup, currGroupSize);
          if (data) {
            data.color = color;
            results.push(data);
          }
        }
      }
    }

    return this.mergeRectangles_(results);
  };

  // Default colors
  //===================

  tracking.ColorTracker.registerColor('cyan', function(r, g, b) {
    var thresholdGreen = 50,
      thresholdBlue = 70,
      dx = r - 0,
      dy = g - 255,
      dz = b - 255;

    if ((g - r) >= thresholdGreen && (b - r) >= thresholdBlue) {
      return true;
    }
    return dx * dx + dy * dy + dz * dz < 6400;
  });

  tracking.ColorTracker.registerColor('magenta', function(r, g, b) {
    var threshold = 50,
      dx = r - 255,
      dy = g - 0,
      dz = b - 255;

    if ((r - g) >= threshold && (b - g) >= threshold) {
      return true;
    }
    return dx * dx + dy * dy + dz * dz < 19600;
  });

  tracking.ColorTracker.registerColor('yellow', function(r, g, b) {
    var threshold = 50,
      dx = r - 255,
      dy = g - 255,
      dz = b - 0;

    if ((r - b) >= threshold && (g - b) >= threshold) {
      return true;
    }
    return dx * dx + dy * dy + dz * dz < 10000;
  });


  // Caching neighbour i/j offset values.
  //=====================================
  var neighboursI = new Int32Array([-1, -1, 0, 1, 1, 1, 0, -1]);
  var neighboursJ = new Int32Array([0, 1, 1, 1, 0, -1, -1, -1]);
}());

(function() {
  /**
   * ObjectTracker utility.
   * @constructor
   * @param {string|Array.<string|Array.<number>>} opt_classifiers Optional
   *     object classifiers to track.
   * @extends {tracking.Tracker}
   */
  tracking.ObjectTracker = function(opt_classifiers) {
    tracking.ObjectTracker.base(this, 'constructor');

    if (opt_classifiers) {
      if (!Array.isArray(opt_classifiers)) {
        opt_classifiers = [opt_classifiers];
      }

      if (Array.isArray(opt_classifiers)) {
        opt_classifiers.forEach(function(classifier, i) {
          if (typeof classifier === 'string') {
            opt_classifiers[i] = tracking.ViolaJones.classifiers[classifier];
          }
          if (!opt_classifiers[i]) {
            throw new Error('Object classifier not valid, try `new tracking.ObjectTracker("face")`.');
          }
        });
      }
    }

    this.setClassifiers(opt_classifiers);
  };

  tracking.inherits(tracking.ObjectTracker, tracking.Tracker);

  /**
   * Specifies the edges density of a block in order to decide whether to skip
   * it or not.
   * @default 0.2
   * @type {number}
   */
  tracking.ObjectTracker.prototype.edgesDensity = 0.2;

  /**
   * Specifies the initial scale to start the feature block scaling.
   * @default 1.0
   * @type {number}
   */
  tracking.ObjectTracker.prototype.initialScale = 1.0;

  /**
   * Specifies the scale factor to scale the feature block.
   * @default 1.25
   * @type {number}
   */
  tracking.ObjectTracker.prototype.scaleFactor = 1.25;

  /**
   * Specifies the block step size.
   * @default 1.5
   * @type {number}
   */
  tracking.ObjectTracker.prototype.stepSize = 1.5;

  /**
   * Gets the tracker HAAR classifiers.
   * @return {TypedArray.<number>}
   */
  tracking.ObjectTracker.prototype.getClassifiers = function() {
    return this.classifiers;
  };

  /**
   * Gets the edges density value.
   * @return {number}
   */
  tracking.ObjectTracker.prototype.getEdgesDensity = function() {
    return this.edgesDensity;
  };

  /**
   * Gets the initial scale to start the feature block scaling.
   * @return {number}
   */
  tracking.ObjectTracker.prototype.getInitialScale = function() {
    return this.initialScale;
  };

  /**
   * Gets the scale factor to scale the feature block.
   * @return {number}
   */
  tracking.ObjectTracker.prototype.getScaleFactor = function() {
    return this.scaleFactor;
  };

  /**
   * Gets the block step size.
   * @return {number}
   */
  tracking.ObjectTracker.prototype.getStepSize = function() {
    return this.stepSize;
  };

  /**
   * Tracks the `Video` frames. This method is called for each video frame in
   * order to emit `track` event.
   * @param {Uint8ClampedArray} pixels The pixels data to track.
   * @param {number} width The pixels canvas width.
   * @param {number} height The pixels canvas height.
   */
  tracking.ObjectTracker.prototype.track = function(pixels, width, height) {
    var self = this;
    var classifiers = this.getClassifiers();

    if (!classifiers) {
      throw new Error('Object classifier not specified, try `new tracking.ObjectTracker("face")`.');
    }

    var results = [];

    classifiers.forEach(function(classifier) {
      results = results.concat(tracking.ViolaJones.detect(pixels, width, height, self.getInitialScale(), self.getScaleFactor(), self.getStepSize(), self.getEdgesDensity(), classifier));
    });

    this.emit('track', {
      data: results
    });
  };

  /**
   * Sets the tracker HAAR classifiers.
   * @param {TypedArray.<number>} classifiers
   */
  tracking.ObjectTracker.prototype.setClassifiers = function(classifiers) {
    this.classifiers = classifiers;
  };

  /**
   * Sets the edges density.
   * @param {number} edgesDensity
   */
  tracking.ObjectTracker.prototype.setEdgesDensity = function(edgesDensity) {
    this.edgesDensity = edgesDensity;
  };

  /**
   * Sets the initial scale to start the block scaling.
   * @param {number} initialScale
   */
  tracking.ObjectTracker.prototype.setInitialScale = function(initialScale) {
    this.initialScale = initialScale;
  };

  /**
   * Sets the scale factor to scale the feature block.
   * @param {number} scaleFactor
   */
  tracking.ObjectTracker.prototype.setScaleFactor = function(scaleFactor) {
    this.scaleFactor = scaleFactor;
  };

  /**
   * Sets the block step size.
   * @param {number} stepSize
   */
  tracking.ObjectTracker.prototype.setStepSize = function(stepSize) {
    this.stepSize = stepSize;
  };

}());

/**
 * tracking.js - A modern approach for Computer Vision on the web.
 * @author Eduardo Lundgren <edu@rdo.io>
 * @version v1.0.0
 * @link http://trackingjs.com
 * @license BSD
 */
tracking.ViolaJones.classifiers.face=new Float64Array([20,20,.822689414024353,3,0,2,3,7,14,4,-1,3,9,14,2,2,.004014195874333382,.0337941907346249,.8378106951713562,0,2,1,2,18,4,-1,7,2,6,4,3,.0151513395830989,.1514132022857666,.7488812208175659,0,2,1,7,15,9,-1,1,10,15,3,3,.004210993181914091,.0900492817163467,.6374819874763489,6.956608772277832,16,0,2,5,6,2,6,-1,5,9,2,3,2,.0016227109590545297,.0693085864186287,.7110946178436279,0,2,7,5,6,3,-1,9,5,2,3,3,.002290664939209819,.1795803010463715,.6668692231178284,0,2,4,0,12,9,-1,4,3,12,3,3,.005002570804208517,.1693672984838486,.6554006934165955,0,2,6,9,10,8,-1,6,13,10,4,2,.007965989410877228,.5866332054138184,.0914145186543465,0,2,3,6,14,8,-1,3,10,14,4,2,-.003522701095789671,.1413166970014572,.6031895875930786,0,2,14,1,6,10,-1,14,1,3,10,2,.0366676896810532,.3675672113895416,.7920318245887756,0,2,7,8,5,12,-1,7,12,5,4,3,.009336147457361221,.6161385774612427,.2088509947061539,0,2,1,1,18,3,-1,7,1,6,3,3,.008696131408214569,.2836230993270874,.6360273957252502,0,2,1,8,17,2,-1,1,9,17,1,2,.0011488880263641477,.2223580926656723,.5800700783729553,0,2,16,6,4,2,-1,16,7,4,1,2,-.002148468978703022,.2406464070081711,.5787054896354675,0,2,5,17,2,2,-1,5,18,2,1,2,.002121906029060483,.5559654831886292,.136223703622818,0,2,14,2,6,12,-1,14,2,3,12,2,-.0939491465687752,.8502737283706665,.4717740118503571,0,3,4,0,4,12,-1,4,0,2,6,2,6,6,2,6,2,.0013777789426967502,.5993673801422119,.2834529876708984,0,2,2,11,18,8,-1,8,11,6,8,3,.0730631574988365,.4341886043548584,.7060034275054932,0,2,5,7,10,2,-1,5,8,10,1,2,.00036767389974556863,.3027887940406799,.6051574945449829,0,2,15,11,5,3,-1,15,12,5,1,3,-.0060479710809886456,.17984339594841,.5675256848335266,9.498542785644531,21,0,2,5,3,10,9,-1,5,6,10,3,3,-.0165106896311045,.6644225120544434,.1424857974052429,0,2,9,4,2,14,-1,9,11,2,7,2,.002705249935388565,.6325352191925049,.1288477033376694,0,2,3,5,4,12,-1,3,9,4,4,3,.002806986914947629,.1240288019180298,.6193193197250366,0,2,4,5,12,5,-1,8,5,4,5,3,-.0015402400167658925,.1432143002748489,.5670015811920166,0,2,5,6,10,8,-1,5,10,10,4,2,-.0005638627917505801,.1657433062791824,.5905207991600037,0,2,8,0,6,9,-1,8,3,6,3,3,.0019253729842603207,.2695507109165192,.5738824009895325,0,2,9,12,1,8,-1,9,16,1,4,2,-.005021484103053808,.1893538981676102,.5782774090766907,0,2,0,7,20,6,-1,0,9,20,2,3,.0026365420781075954,.2309329062700272,.5695425868034363,0,2,7,0,6,17,-1,9,0,2,17,3,-.0015127769438549876,.2759602069854736,.5956642031669617,0,2,9,0,6,4,-1,11,0,2,4,3,-.0101574398577213,.1732538044452667,.5522047281265259,0,2,5,1,6,4,-1,7,1,2,4,3,-.011953660286963,.1339409947395325,.5559014081954956,0,2,12,1,6,16,-1,14,1,2,16,3,.004885949194431305,.3628703951835632,.6188849210739136,0,3,0,5,18,8,-1,0,5,9,4,2,9,9,9,4,2,-.0801329165697098,.0912110507488251,.5475944876670837,0,3,8,15,10,4,-1,13,15,5,2,2,8,17,5,2,2,.0010643280111253262,.3715142905712128,.5711399912834167,0,3,3,1,4,8,-1,3,1,2,4,2,5,5,2,4,2,-.0013419450260698795,.5953313708305359,.331809788942337,0,3,3,6,14,10,-1,10,6,7,5,2,3,11,7,5,2,-.0546011403203011,.1844065934419632,.5602846145629883,0,2,2,1,6,16,-1,4,1,2,16,3,.0029071690514683723,.3594244122505188,.6131715178489685,0,2,0,18,20,2,-1,0,19,20,1,2,.0007471871795132756,.5994353294372559,.3459562957286835,0,2,8,13,4,3,-1,8,14,4,1,3,.004301380831748247,.4172652065753937,.6990845203399658,0,2,9,14,2,3,-1,9,15,2,1,3,.004501757211983204,.4509715139865875,.7801457047462463,0,2,0,12,9,6,-1,0,14,9,2,3,.0241385009139776,.5438212752342224,.1319826990365982,18.4129695892334,39,0,2,5,7,3,4,-1,5,9,3,2,2,.001921223010867834,.1415266990661621,.6199870705604553,0,2,9,3,2,16,-1,9,11,2,8,2,-.00012748669541906565,.6191074252128601,.1884928941726685,0,2,3,6,13,8,-1,3,10,13,4,2,.0005140993162058294,.1487396955490112,.5857927799224854,0,2,12,3,8,2,-1,12,3,4,2,2,.004187860991805792,.2746909856796265,.6359239816665649,0,2,8,8,4,12,-1,8,12,4,4,3,.005101571790874004,.5870851278305054,.2175628989934921,0,3,11,3,8,6,-1,15,3,4,3,2,11,6,4,3,2,-.002144844038411975,.5880944728851318,.2979590892791748,0,2,7,1,6,19,-1,9,1,2,19,3,-.0028977119363844395,.2373327016830444,.5876647233963013,0,2,9,0,6,4,-1,11,0,2,4,3,-.0216106791049242,.1220654994249344,.5194202065467834,0,2,3,1,9,3,-1,6,1,3,3,3,-.004629931878298521,.263123095035553,.5817409157752991,0,3,8,15,10,4,-1,13,15,5,2,2,8,17,5,2,2,.000593937118537724,.363862007856369,.5698544979095459,0,2,0,3,6,10,-1,3,3,3,10,2,.0538786612451077,.4303531050682068,.7559366226196289,0,2,3,4,15,15,-1,3,9,15,5,3,.0018887349870055914,.2122603058815002,.561342716217041,0,2,6,5,8,6,-1,6,7,8,2,3,-.0023635339457541704,.563184916973114,.2642767131328583,0,3,4,4,12,10,-1,10,4,6,5,2,4,9,6,5,2,.0240177996456623,.5797107815742493,.2751705944538117,0,2,6,4,4,4,-1,8,4,2,4,2,.00020543030404951423,.2705242037773132,.575256884098053,0,2,15,11,1,2,-1,15,12,1,1,2,.0008479019743390381,.5435624718666077,.2334876954555512,0,2,3,11,2,2,-1,3,12,2,1,2,.0014091329649090767,.5319424867630005,.2063155025243759,0,2,16,11,1,3,-1,16,12,1,1,3,.0014642629539594054,.5418980717658997,.3068861067295075,0,3,3,15,6,4,-1,3,15,3,2,2,6,17,3,2,2,.0016352549428120255,.3695372939109802,.6112868189811707,0,2,6,7,8,2,-1,6,8,8,1,2,.0008317275205627084,.3565036952495575,.6025236248970032,0,2,3,11,1,3,-1,3,12,1,1,3,-.0020998890977352858,.1913982033729553,.5362827181816101,0,2,6,0,12,2,-1,6,1,12,1,2,-.0007421398186124861,.3835555016994476,.552931010723114,0,2,9,14,2,3,-1,9,15,2,1,3,.0032655049581080675,.4312896132469177,.7101895809173584,0,2,7,15,6,2,-1,7,16,6,1,2,.0008913499186746776,.3984830975532532,.6391963958740234,0,2,0,5,4,6,-1,0,7,4,2,3,-.0152841797098517,.2366732954978943,.5433713793754578,0,2,4,12,12,2,-1,8,12,4,2,3,.004838141147047281,.5817500948905945,.3239189088344574,0,2,6,3,1,9,-1,6,6,1,3,3,-.0009109317907132208,.5540593862533569,.2911868989467621,0,2,10,17,3,2,-1,11,17,1,2,3,-.006127506028860807,.1775255054235458,.5196629166603088,0,2,9,9,2,2,-1,9,10,2,1,2,-.00044576259097084403,.3024170100688934,.5533593893051147,0,2,7,6,6,4,-1,9,6,2,4,3,.0226465407758951,.4414930939674377,.6975377202033997,0,2,7,17,3,2,-1,8,17,1,2,3,-.0018804960418492556,.2791394889354706,.5497952103614807,0,2,10,17,3,3,-1,11,17,1,3,3,.007088910788297653,.5263199210166931,.2385547012090683,0,2,8,12,3,2,-1,8,13,3,1,2,.0017318050377070904,.4319379031658173,.6983600854873657,0,2,9,3,6,2,-1,11,3,2,2,3,-.006848270073533058,.3082042932510376,.5390920042991638,0,2,3,11,14,4,-1,3,13,14,2,2,-15062530110299122e-21,.552192211151123,.3120366036891937,0,3,1,10,18,4,-1,10,10,9,2,2,1,12,9,2,2,.0294755697250366,.5401322841644287,.1770603060722351,0,2,0,10,3,3,-1,0,11,3,1,3,.008138732984662056,.5178617835044861,.121101900935173,0,2,9,1,6,6,-1,11,1,2,6,3,.0209429506212473,.5290294289588928,.3311221897602081,0,2,8,7,3,6,-1,9,7,1,6,3,-.009566552937030792,.7471994161605835,.4451968967914581,15.324139595031738,33,0,2,1,0,18,9,-1,1,3,18,3,3,-.00028206960996612906,.2064086049795151,.6076732277870178,0,2,12,10,2,6,-1,12,13,2,3,2,.00167906004935503,.5851997137069702,.1255383938550949,0,2,0,5,19,8,-1,0,9,19,4,2,.0006982791237533092,.094018429517746,.5728961229324341,0,2,7,0,6,9,-1,9,0,2,9,3,.0007895901217125356,.1781987994909287,.5694308876991272,0,2,5,3,6,1,-1,7,3,2,1,3,-.002856049919500947,.1638399064540863,.5788664817810059,0,2,11,3,6,1,-1,13,3,2,1,3,-.0038122469559311867,.2085440009832382,.5508564710617065,0,2,5,10,4,6,-1,5,13,4,3,2,.0015896620461717248,.5702760815620422,.1857215017080307,0,2,11,3,6,1,-1,13,3,2,1,3,.0100783398374915,.5116943120956421,.2189770042896271,0,2,4,4,12,6,-1,4,6,12,2,3,-.0635263025760651,.7131379842758179,.4043813049793243,0,2,15,12,2,6,-1,15,14,2,2,3,-.009103149175643921,.2567181885242462,.54639732837677,0,2,9,3,2,2,-1,10,3,1,2,2,-.002403500024229288,.1700665950775147,.559097409248352,0,2,9,3,3,1,-1,10,3,1,1,3,.001522636041045189,.5410556793212891,.2619054019451141,0,2,1,1,4,14,-1,3,1,2,14,2,.0179974399507046,.3732436895370483,.6535220742225647,0,3,9,0,4,4,-1,11,0,2,2,2,9,2,2,2,2,-.00645381910726428,.2626481950283051,.5537446141242981,0,2,7,5,1,14,-1,7,12,1,7,2,-.0118807600811124,.2003753930330277,.5544745922088623,0,2,19,0,1,4,-1,19,2,1,2,2,.0012713660253211856,.5591902732849121,.303197592496872,0,2,5,5,6,4,-1,8,5,3,4,2,.0011376109905540943,.2730407118797302,.5646508932113647,0,2,9,18,3,2,-1,10,18,1,2,3,-.00426519988104701,.1405909061431885,.5461820960044861,0,2,8,18,3,2,-1,9,18,1,2,3,-.0029602861031889915,.1795035004615784,.5459290146827698,0,2,4,5,12,6,-1,4,7,12,2,3,-.008844822645187378,.5736783146858215,.280921995639801,0,2,3,12,2,6,-1,3,14,2,2,3,-.006643068976700306,.2370675951242447,.5503826141357422,0,2,10,8,2,12,-1,10,12,2,4,3,.003999780863523483,.5608199834823608,.3304282128810883,0,2,7,18,3,2,-1,8,18,1,2,3,-.004122172016650438,.1640105992555618,.5378993153572083,0,2,9,0,6,2,-1,11,0,2,2,3,.0156249096617103,.5227649211883545,.2288603931665421,0,2,5,11,9,3,-1,5,12,9,1,3,-.0103564197197557,.7016193866729736,.4252927899360657,0,2,9,0,6,2,-1,11,0,2,2,3,-.008796080946922302,.2767347097396851,.5355830192565918,0,2,1,1,18,5,-1,7,1,6,5,3,.1622693985700607,.434224009513855,.744257926940918,0,3,8,0,4,4,-1,10,0,2,2,2,8,2,2,2,2,.0045542530715465546,.5726485848426819,.2582125067710877,0,2,3,12,1,3,-1,3,13,1,1,3,-.002130920998752117,.2106848061084747,.5361018776893616,0,2,8,14,5,3,-1,8,15,5,1,3,-.0132084200158715,.7593790888786316,.4552468061447144,0,3,5,4,10,12,-1,5,4,5,6,2,10,10,5,6,2,-.0659966766834259,.125247597694397,.5344039797782898,0,2,9,6,9,12,-1,9,10,9,4,3,.007914265617728233,.3315384089946747,.5601043105125427,0,3,2,2,12,14,-1,2,2,6,7,2,8,9,6,7,2,.0208942797034979,.5506049990653992,.2768838107585907,21.010639190673828,44,0,2,4,7,12,2,-1,8,7,4,2,3,.0011961159761995077,.1762690991163254,.6156241297721863,0,2,7,4,6,4,-1,7,6,6,2,2,-.0018679830245673656,.6118106842041016,.1832399964332581,0,2,4,5,11,8,-1,4,9,11,4,2,-.00019579799845814705,.0990442633628845,.5723816156387329,0,2,3,10,16,4,-1,3,12,16,2,2,-.0008025565766729414,.5579879879951477,.2377282977104187,0,2,0,0,16,2,-1,0,1,16,1,2,-.0024510810617357492,.2231457978487015,.5858935117721558,0,2,7,5,6,2,-1,9,5,2,2,3,.0005036185029894114,.2653993964195252,.5794103741645813,0,3,3,2,6,10,-1,3,2,3,5,2,6,7,3,5,2,.0040293349884450436,.5803827047348022,.2484865039587021,0,2,10,5,8,15,-1,10,10,8,5,3,-.0144517095759511,.1830351948738098,.5484204888343811,0,3,3,14,8,6,-1,3,14,4,3,2,7,17,4,3,2,.0020380979403853416,.3363558948040009,.6051092743873596,0,2,14,2,2,2,-1,14,3,2,1,2,-.0016155190533027053,.2286642044782639,.5441246032714844,0,2,1,10,7,6,-1,1,13,7,3,2,.0033458340913057327,.5625913143157959,.2392338067293167,0,2,15,4,4,3,-1,15,4,2,3,2,.0016379579901695251,.3906993865966797,.5964621901512146,0,3,2,9,14,6,-1,2,9,7,3,2,9,12,7,3,2,.0302512105554342,.524848222732544,.1575746983289719,0,2,5,7,10,4,-1,5,9,10,2,2,.037251990288496,.4194310903549194,.6748418807983398,0,3,6,9,8,8,-1,6,9,4,4,2,10,13,4,4,2,-.0251097902655602,.1882549971342087,.5473451018333435,0,2,14,1,3,2,-1,14,2,3,1,2,-.005309905856847763,.133997306227684,.5227110981941223,0,2,1,4,4,2,-1,3,4,2,2,2,.0012086479691788554,.3762088119983673,.6109635829925537,0,2,11,10,2,8,-1,11,14,2,4,2,-.0219076797366142,.266314297914505,.5404006838798523,0,2,0,0,5,3,-1,0,1,5,1,3,.0054116579703986645,.5363578796386719,.2232273072004318,0,3,2,5,18,8,-1,11,5,9,4,2,2,9,9,4,2,.069946326315403,.5358232855796814,.2453698068857193,0,2,6,6,1,6,-1,6,9,1,3,2,.00034520021290518343,.2409671992063522,.5376930236816406,0,2,19,1,1,3,-1,19,2,1,1,3,.0012627709656953812,.5425856709480286,.3155693113803864,0,2,7,6,6,6,-1,9,6,2,6,3,.0227195098996162,.4158405959606171,.6597865223884583,0,2,19,1,1,3,-1,19,2,1,1,3,-.001811100053600967,.2811253070831299,.5505244731903076,0,2,3,13,2,3,-1,3,14,2,1,3,.0033469670452177525,.526002824306488,.1891465038061142,0,3,8,4,8,12,-1,12,4,4,6,2,8,10,4,6,2,.00040791751234792173,.5673509240150452,.3344210088253021,0,2,5,2,6,3,-1,7,2,2,3,3,.0127347996458411,.5343592166900635,.2395612001419067,0,2,6,1,9,10,-1,6,6,9,5,2,-.007311972789466381,.6010890007019043,.4022207856178284,0,2,0,4,6,12,-1,2,4,2,12,3,-.0569487512111664,.8199151158332825,.4543190896511078,0,2,15,13,2,3,-1,15,14,2,1,3,-.005011659115552902,.2200281023979187,.5357710719108582,0,2,7,14,5,3,-1,7,15,5,1,3,.006033436860889196,.4413081109523773,.7181751132011414,0,2,15,13,3,3,-1,15,14,3,1,3,.0039437441155314445,.547886073589325,.2791733145713806,0,2,6,14,8,3,-1,6,15,8,1,3,-.0036591119132936,.635786771774292,.3989723920822144,0,2,15,13,3,3,-1,15,14,3,1,3,-.0038456181064248085,.3493686020374298,.5300664901733398,0,2,2,13,3,3,-1,2,14,3,1,3,-.007192626129835844,.1119614988565445,.5229672789573669,0,3,4,7,12,12,-1,10,7,6,6,2,4,13,6,6,2,-.0527989417314529,.2387102991342545,.54534512758255,0,2,9,7,2,6,-1,10,7,1,6,2,-.007953766733407974,.7586917877197266,.4439376890659332,0,2,8,9,5,2,-1,8,10,5,1,2,-.0027344180271029472,.2565476894378662,.5489321947097778,0,2,8,6,3,4,-1,9,6,1,4,3,-.0018507939530536532,.6734347939491272,.4252474904060364,0,2,9,6,2,8,-1,9,10,2,4,2,.0159189198166132,.548835277557373,.2292661964893341,0,2,7,7,3,6,-1,8,7,1,6,3,-.0012687679845839739,.6104331016540527,.4022389948368073,0,2,11,3,3,3,-1,12,3,1,3,3,.006288391072303057,.5310853123664856,.1536193042993546,0,2,5,4,6,1,-1,7,4,2,1,3,-.0062259892001748085,.1729111969470978,.524160623550415,0,2,5,6,10,3,-1,5,7,10,1,3,-.0121325999498367,.659775972366333,.4325182139873505,23.918790817260742,50,0,2,7,3,6,9,-1,7,6,6,3,3,-.0039184908382594585,.6103435158729553,.1469330936670303,0,2,6,7,9,1,-1,9,7,3,1,3,.0015971299726516008,.2632363140583038,.5896466970443726,0,2,2,8,16,8,-1,2,12,16,4,2,.0177801102399826,.587287425994873,.1760361939668655,0,2,14,6,2,6,-1,14,9,2,3,2,.0006533476989716291,.1567801982164383,.5596066117286682,0,2,1,5,6,15,-1,1,10,6,5,3,-.00028353091329336166,.1913153976202011,.5732036232948303,0,2,10,0,6,9,-1,10,3,6,3,3,.0016104689566418529,.2914913892745972,.5623080730438232,0,2,6,6,7,14,-1,6,13,7,7,2,-.0977506190538406,.194347694516182,.5648233294487,0,2,13,7,3,6,-1,13,9,3,2,3,.0005518235848285258,.3134616911411285,.5504639744758606,0,2,1,8,15,4,-1,6,8,5,4,3,-.0128582203760743,.253648191690445,.5760142803192139,0,2,11,2,3,10,-1,11,7,3,5,2,.004153023939579725,.5767722129821777,.36597740650177,0,2,3,7,4,6,-1,3,9,4,2,3,.0017092459602281451,.2843191027641296,.5918939113616943,0,2,13,3,6,10,-1,15,3,2,10,3,.007521735969930887,.4052427113056183,.6183109283447266,0,3,5,7,8,10,-1,5,7,4,5,2,9,12,4,5,2,.0022479810286313295,.578375518321991,.3135401010513306,0,3,4,4,12,12,-1,10,4,6,6,2,4,10,6,6,2,.0520062111318111,.5541312098503113,.1916636973619461,0,2,1,4,6,9,-1,3,4,2,9,3,.0120855299755931,.4032655954360962,.6644591093063354,0,2,11,3,2,5,-1,11,3,1,5,2,14687820112158079e-21,.3535977900028229,.5709382891654968,0,2,7,3,2,5,-1,8,3,1,5,2,7139518857002258e-21,.3037444949150085,.5610269904136658,0,2,10,14,2,3,-1,10,15,2,1,3,-.0046001640148460865,.7181087136268616,.4580326080322266,0,2,5,12,6,2,-1,8,12,3,2,2,.0020058949012309313,.5621951818466187,.2953684031963348,0,2,9,14,2,3,-1,9,15,2,1,3,.004505027085542679,.4615387916564941,.7619017958641052,0,2,4,11,12,6,-1,4,14,12,3,2,.0117468303069472,.5343837141990662,.1772529035806656,0,2,11,11,5,9,-1,11,14,5,3,3,-.0583163388073444,.1686245948076248,.5340772271156311,0,2,6,15,3,2,-1,6,16,3,1,2,.00023629379575140774,.3792056143283844,.6026803851127625,0,2,11,0,3,5,-1,12,0,1,5,3,-.007815618067979813,.151286706328392,.5324323773384094,0,2,5,5,6,7,-1,8,5,3,7,2,-.0108761601150036,.2081822007894516,.5319945216178894,0,2,13,0,1,9,-1,13,3,1,3,3,-.0027745519764721394,.4098246991634369,.5210328102111816,0,3,3,2,4,8,-1,3,2,2,4,2,5,6,2,4,2,-.0007827638182789087,.5693274140357971,.3478842079639435,0,2,13,12,4,6,-1,13,14,4,2,3,.0138704096898437,.5326750874519348,.2257698029279709,0,2,3,12,4,6,-1,3,14,4,2,3,-.0236749108880758,.1551305055618286,.5200707912445068,0,2,13,11,3,4,-1,13,13,3,2,2,-14879409718560055e-21,.5500566959381104,.3820176124572754,0,2,4,4,4,3,-1,4,5,4,1,3,.00361906411126256,.4238683879375458,.6639748215675354,0,2,7,5,11,8,-1,7,9,11,4,2,-.0198171101510525,.2150038033723831,.5382357835769653,0,2,7,8,3,4,-1,8,8,1,4,3,-.0038154039066284895,.6675711274147034,.4215297102928162,0,2,9,1,6,1,-1,11,1,2,1,3,-.0049775829538702965,.2267289012670517,.5386328101158142,0,2,5,5,3,3,-1,5,6,3,1,3,.002244102070108056,.4308691024780273,.6855735778808594,0,3,0,9,20,6,-1,10,9,10,3,2,0,12,10,3,2,.0122824599966407,.5836614966392517,.3467479050159454,0,2,8,6,3,5,-1,9,6,1,5,3,-.002854869933798909,.7016944885253906,.4311453998088837,0,2,11,0,1,3,-1,11,1,1,1,3,-.0037875669077038765,.2895345091819763,.5224946141242981,0,2,4,2,4,2,-1,4,3,4,1,2,-.0012201230274513364,.2975570857524872,.5481644868850708,0,2,12,6,4,3,-1,12,7,4,1,3,.010160599835217,.4888817965984345,.8182697892189026,0,2,5,0,6,4,-1,7,0,2,4,3,-.0161745697259903,.1481492966413498,.5239992737770081,0,2,9,7,3,8,-1,10,7,1,8,3,.0192924607545137,.4786309897899628,.7378190755844116,0,2,9,7,2,2,-1,10,7,1,2,2,-.003247953951358795,.7374222874641418,.4470643997192383,0,3,6,7,14,4,-1,13,7,7,2,2,6,9,7,2,2,-.009380348026752472,.3489154875278473,.5537996292114258,0,2,0,5,3,6,-1,0,7,3,2,3,-.0126061299815774,.2379686981439591,.5315443277359009,0,2,13,11,3,4,-1,13,13,3,2,2,-.0256219301372766,.1964688003063202,.5138769745826721,0,2,4,11,3,4,-1,4,13,3,2,2,-7574149640277028e-20,.5590522885322571,.3365853130817413,0,3,5,9,12,8,-1,11,9,6,4,2,5,13,6,4,2,-.0892108827829361,.0634046569466591,.516263484954834,0,2,9,12,1,3,-1,9,13,1,1,3,-.002767048077657819,.732346773147583,.4490706026554108,0,2,10,15,2,4,-1,10,17,2,2,2,.0002715257869567722,.411483496427536,.5985518097877502,24.52787971496582,51,0,2,7,7,6,1,-1,9,7,2,1,3,.001478621968999505,.266354501247406,.6643316745758057,0,3,12,3,6,6,-1,15,3,3,3,2,12,6,3,3,2,-.001874165958724916,.6143848896026611,.2518512904644013,0,2,0,4,10,6,-1,0,6,10,2,3,-.001715100952424109,.5766341090202332,.2397463023662567,0,3,8,3,8,14,-1,12,3,4,7,2,8,10,4,7,2,-.0018939269939437509,.5682045817375183,.2529144883155823,0,2,4,4,7,15,-1,4,9,7,5,3,-.005300605203956366,.1640675961971283,.5556079745292664,0,3,12,2,6,8,-1,15,2,3,4,2,12,6,3,4,2,-.0466625317931175,.6123154163360596,.4762830138206482,0,3,2,2,6,8,-1,2,2,3,4,2,5,6,3,4,2,-.000794313324149698,.5707858800888062,.2839404046535492,0,2,2,13,18,7,-1,8,13,6,7,3,.0148916700854898,.4089672863483429,.6006367206573486,0,3,4,3,8,14,-1,4,3,4,7,2,8,10,4,7,2,-.0012046529445797205,.5712450742721558,.2705289125442505,0,2,18,1,2,6,-1,18,3,2,2,3,.006061938125640154,.526250422000885,.3262225985527039,0,2,9,11,2,3,-1,9,12,2,1,3,-.0025286648888140917,.6853830814361572,.4199256896972656,0,2,18,1,2,6,-1,18,3,2,2,3,-.005901021882891655,.3266282081604004,.5434812903404236,0,2,0,1,2,6,-1,0,3,2,2,3,.005670276004821062,.5468410849571228,.2319003939628601,0,2,1,5,18,6,-1,1,7,18,2,3,-.003030410036444664,.557066798210144,.2708238065242767,0,2,0,2,6,7,-1,3,2,3,7,2,.002980364952236414,.3700568974018097,.5890625715255737,0,2,7,3,6,14,-1,7,10,6,7,2,-.0758405104279518,.2140070050954819,.5419948101043701,0,2,3,7,13,10,-1,3,12,13,5,2,.0192625392228365,.5526772141456604,.2726590037345886,0,2,11,15,2,2,-1,11,16,2,1,2,.00018888259364757687,.3958011865615845,.6017209887504578,0,3,2,11,16,4,-1,2,11,8,2,2,10,13,8,2,2,.0293695498257875,.5241373777389526,.1435758024454117,0,3,13,7,6,4,-1,16,7,3,2,2,13,9,3,2,2,.0010417619487270713,.3385409116744995,.5929983258247375,0,2,6,10,3,9,-1,6,13,3,3,3,.0026125640142709017,.5485377907752991,.3021597862243652,0,2,14,6,1,6,-1,14,9,1,3,2,.0009697746718302369,.3375276029109955,.553203284740448,0,2,5,10,4,1,-1,7,10,2,1,2,.0005951265920884907,.563174307346344,.3359399139881134,0,2,3,8,15,5,-1,8,8,5,5,3,-.1015655994415283,.0637350380420685,.5230425000190735,0,2,1,6,5,4,-1,1,8,5,2,2,.0361566990613937,.5136963129043579,.1029528975486755,0,2,3,1,17,6,-1,3,3,17,2,3,.003462414024397731,.3879320025444031,.5558289289474487,0,2,6,7,8,2,-1,10,7,4,2,2,.0195549800992012,.5250086784362793,.1875859946012497,0,2,9,7,3,2,-1,10,7,1,2,3,-.0023121440317481756,.667202889919281,.4679641127586365,0,2,8,7,3,2,-1,9,7,1,2,3,-.001860528951510787,.7163379192352295,.4334670901298523,0,2,8,9,4,2,-1,8,10,4,1,2,-.0009402636205777526,.302136093378067,.5650203227996826,0,2,8,8,4,3,-1,8,9,4,1,3,-.005241833161562681,.1820009052753449,.5250256061553955,0,2,9,5,6,4,-1,9,5,3,4,2,.00011729019752237946,.3389188051223755,.544597327709198,0,2,8,13,4,3,-1,8,14,4,1,3,.0011878840159624815,.4085349142551422,.6253563165664673,0,3,4,7,12,6,-1,10,7,6,3,2,4,10,6,3,2,-.0108813596889377,.3378399014472961,.5700082778930664,0,2,8,14,4,3,-1,8,15,4,1,3,.0017354859737679362,.4204635918140411,.6523038744926453,0,2,9,7,3,3,-1,9,8,3,1,3,-.00651190523058176,.2595216035842896,.5428143739700317,0,2,7,4,3,8,-1,8,4,1,8,3,-.0012136430013924837,.6165143847465515,.3977893888950348,0,2,10,0,3,6,-1,11,0,1,6,3,-.010354240424931,.1628028005361557,.5219504833221436,0,2,6,3,4,8,-1,8,3,2,8,2,.0005585883045569062,.3199650943279266,.5503574013710022,0,2,14,3,6,13,-1,14,3,3,13,2,.0152996499091387,.4103994071483612,.6122388243675232,0,2,8,13,3,6,-1,8,16,3,3,2,-.021588210016489,.103491298854351,.519738495349884,0,2,14,3,6,13,-1,14,3,3,13,2,-.1283462941646576,.8493865132331848,.4893102943897247,0,3,0,7,10,4,-1,0,7,5,2,2,5,9,5,2,2,-.0022927189711481333,.3130157887935638,.5471575260162354,0,2,14,3,6,13,-1,14,3,3,13,2,.0799151062965393,.4856320917606354,.6073989272117615,0,2,0,3,6,13,-1,3,3,3,13,2,-.0794410929083824,.8394674062728882,.462453305721283,0,2,9,1,4,1,-1,9,1,2,1,2,-.00528000108897686,.1881695985794067,.5306698083877563,0,2,8,0,2,1,-1,9,0,1,1,2,.0010463109938427806,.5271229147911072,.2583065927028656,0,3,10,16,4,4,-1,12,16,2,2,2,10,18,2,2,2,.00026317298761568964,.4235304892063141,.5735440850257874,0,2,9,6,2,3,-1,10,6,1,3,2,-.0036173160187900066,.6934396028518677,.4495444893836975,0,2,4,5,12,2,-1,8,5,4,2,3,.0114218797534704,.590092122554779,.4138193130493164,0,2,8,7,3,5,-1,9,7,1,5,3,-.0019963278900831938,.6466382741928101,.4327239990234375,27.153350830078125,56,0,2,6,4,8,6,-1,6,6,8,2,3,-.00996912457048893,.6142324209213257,.2482212036848068,0,2,9,5,2,12,-1,9,11,2,6,2,.0007307305932044983,.5704951882362366,.2321965992450714,0,2,4,6,6,8,-1,4,10,6,4,2,.0006404530140571296,.2112251967191696,.5814933180809021,0,2,12,2,8,5,-1,12,2,4,5,2,.004542401991784573,.2950482070446014,.586631178855896,0,2,0,8,18,3,-1,0,9,18,1,3,9247744310414419e-20,.2990990877151489,.5791326761245728,0,2,8,12,4,8,-1,8,16,4,4,2,-.008660314604640007,.2813029885292053,.5635542273521423,0,2,0,2,8,5,-1,4,2,4,5,2,.008051581680774689,.3535369038581848,.6054757237434387,0,2,13,11,3,4,-1,13,13,3,2,2,.00043835240649059415,.5596532225608826,.2731510996818543,0,2,5,11,6,1,-1,7,11,2,1,3,-981689736363478e-19,.5978031754493713,.3638561069965363,0,2,11,3,3,1,-1,12,3,1,1,3,-.0011298790341243148,.2755252122879028,.5432729125022888,0,2,7,13,5,3,-1,7,14,5,1,3,.006435615010559559,.4305641949176788,.7069833278656006,0,2,11,11,7,6,-1,11,14,7,3,2,-.0568293295800686,.2495242953300476,.5294997096061707,0,2,2,11,7,6,-1,2,14,7,3,2,.004066816996783018,.5478553175926208,.2497723996639252,0,2,12,14,2,6,-1,12,16,2,2,3,481647984997835e-19,.3938601016998291,.5706356167793274,0,2,8,14,3,3,-1,8,15,3,1,3,.00617950176820159,.440760612487793,.7394766807556152,0,2,11,0,3,5,-1,12,0,1,5,3,.006498575210571289,.5445243120193481,.2479152977466583,0,2,6,1,4,9,-1,8,1,2,9,2,-.0010211090557277203,.2544766962528229,.5338971018791199,0,2,10,3,6,1,-1,12,3,2,1,3,-.005424752831459045,.2718858122825623,.5324069261550903,0,2,8,8,3,4,-1,8,10,3,2,2,-.0010559899965301156,.3178288042545319,.553450882434845,0,2,8,12,4,2,-1,8,13,4,1,2,.0006646580877713859,.4284219145774841,.6558194160461426,0,2,5,18,4,2,-1,5,19,4,1,2,-.00027524109464138746,.5902860760688782,.3810262978076935,0,2,2,1,18,6,-1,2,3,18,2,3,.004229320213198662,.381648987531662,.5709385871887207,0,2,6,0,3,2,-1,7,0,1,2,3,-.0032868210691958666,.1747743934392929,.5259544253349304,0,3,13,8,6,2,-1,16,8,3,1,2,13,9,3,1,2,.0001561187964398414,.3601722121238709,.5725612044334412,0,2,6,10,3,6,-1,6,13,3,3,2,-7362138148891972e-21,.540185809135437,.3044497072696686,0,3,0,13,20,4,-1,10,13,10,2,2,0,15,10,2,2,-.014767250046134,.3220770061016083,.5573434829711914,0,2,7,7,6,5,-1,9,7,2,5,3,.0244895908981562,.4301528036594391,.6518812775611877,0,2,11,0,2,2,-1,11,1,2,1,2,-.000376520911231637,.356458306312561,.5598236918449402,0,3,1,8,6,2,-1,1,8,3,1,2,4,9,3,1,2,736576885174145e-20,.3490782976150513,.556189775466919,0,3,0,2,20,2,-1,10,2,10,1,2,0,3,10,1,2,-.0150999398902059,.1776272058486939,.5335299968719482,0,2,7,14,5,3,-1,7,15,5,1,3,-.0038316650316119194,.6149687767028809,.4221394062042236,0,3,7,13,6,6,-1,10,13,3,3,2,7,16,3,3,2,.0169254001230001,.5413014888763428,.2166585028171539,0,2,9,12,2,3,-1,9,13,2,1,3,-.003047785023227334,.6449490785598755,.4354617893695831,0,2,16,11,1,6,-1,16,13,1,2,3,.003214058931916952,.5400155186653137,.3523217141628265,0,2,3,11,1,6,-1,3,13,1,2,3,-.004002320114523172,.2774524092674255,.5338417291641235,0,3,4,4,14,12,-1,11,4,7,6,2,4,10,7,6,2,.0074182129465043545,.567673921585083,.3702817857265472,0,2,5,4,3,3,-1,5,5,3,1,3,-.008876458741724491,.7749221920967102,.4583688974380493,0,2,12,3,3,3,-1,13,3,1,3,3,.002731173997744918,.5338721871376038,.3996661007404327,0,2,6,6,8,3,-1,6,7,8,1,3,-.0025082379579544067,.5611963272094727,.377749890089035,0,2,12,3,3,3,-1,13,3,1,3,3,-.008054107427597046,.291522890329361,.5179182887077332,0,3,3,1,4,10,-1,3,1,2,5,2,5,6,2,5,2,-.0009793881326913834,.5536432862281799,.3700192868709564,0,2,5,7,10,2,-1,5,7,5,2,2,-.005874590948224068,.3754391074180603,.5679376125335693,0,2,8,7,3,3,-1,9,7,1,3,3,-.00449367193505168,.7019699215888977,.4480949938297272,0,2,15,12,2,3,-1,15,13,2,1,3,-.00543892290443182,.2310364991426468,.5313386917114258,0,2,7,8,3,4,-1,8,8,1,4,3,-.0007509464048780501,.5864868760108948,.4129343032836914,0,2,13,4,1,12,-1,13,10,1,6,2,14528800420521293e-21,.3732407093048096,.5619621276855469,0,3,4,5,12,12,-1,4,5,6,6,2,10,11,6,6,2,.0407580696046352,.5312091112136841,.2720521986484528,0,2,7,14,7,3,-1,7,15,7,1,3,.006650593131780624,.4710015952587128,.6693493723869324,0,2,3,12,2,3,-1,3,13,2,1,3,.0045759351924061775,.5167819261550903,.1637275964021683,0,3,3,2,14,2,-1,10,2,7,1,2,3,3,7,1,2,.0065269311890006065,.5397608876228333,.2938531935214996,0,2,0,1,3,10,-1,1,1,1,10,3,-.0136603796854615,.7086488008499146,.453220009803772,0,2,9,0,6,5,-1,11,0,2,5,3,.0273588690906763,.5206481218338013,.3589231967926025,0,2,5,7,6,2,-1,8,7,3,2,2,.0006219755159690976,.3507075905799866,.5441123247146606,0,2,7,1,6,10,-1,7,6,6,5,2,-.0033077080734074116,.5859522819519043,.402489185333252,0,2,1,1,18,3,-1,7,1,6,3,3,-.0106311095878482,.6743267178535461,.4422602951526642,0,2,16,3,3,6,-1,16,5,3,2,3,.0194416493177414,.5282716155052185,.1797904968261719,34.55411148071289,71,0,2,6,3,7,6,-1,6,6,7,3,2,-.005505216773599386,.5914731025695801,.2626559138298035,0,2,4,7,12,2,-1,8,7,4,2,3,.001956227933987975,.2312581986188889,.5741627216339111,0,2,0,4,17,10,-1,0,9,17,5,2,-.008892478421330452,.1656530052423477,.5626654028892517,0,2,3,4,15,16,-1,3,12,15,8,2,.0836383774876595,.5423449873924255,.1957294940948486,0,2,7,15,6,4,-1,7,17,6,2,2,.0012282270472496748,.3417904078960419,.5992503762245178,0,2,15,2,4,9,-1,15,2,2,9,2,.0057629169896245,.3719581961631775,.6079903841018677,0,2,2,3,3,2,-1,2,4,3,1,2,-.0016417410224676132,.2577486038208008,.5576915740966797,0,2,13,6,7,9,-1,13,9,7,3,3,.0034113149158656597,.2950749099254608,.5514171719551086,0,2,8,11,4,3,-1,8,12,4,1,3,-.0110693201422691,.7569358944892883,.4477078914642334,0,3,0,2,20,6,-1,10,2,10,3,2,0,5,10,3,2,.0348659716546535,.5583708882331848,.2669621109962463,0,3,3,2,6,10,-1,3,2,3,5,2,6,7,3,5,2,.0006570109981112182,.5627313256263733,.2988890111446381,0,2,13,10,3,4,-1,13,12,3,2,2,-.0243391301482916,.2771185040473938,.5108863115310669,0,2,4,10,3,4,-1,4,12,3,2,2,.0005943520227447152,.5580651760101318,.3120341897010803,0,2,7,5,6,3,-1,9,5,2,3,3,.0022971509024500847,.3330250084400177,.5679075717926025,0,2,7,6,6,8,-1,7,10,6,4,2,-.0037801829166710377,.2990534901618958,.5344808101654053,0,2,0,11,20,6,-1,0,14,20,3,2,-.13420669734478,.1463858932256699,.5392568111419678,0,3,4,13,4,6,-1,4,13,2,3,2,6,16,2,3,2,.0007522454834543169,.3746953904628754,.5692734718322754,0,3,6,0,8,12,-1,10,0,4,6,2,6,6,4,6,2,-.040545541793108,.2754747867584229,.5484297871589661,0,2,2,0,15,2,-1,2,1,15,1,2,.0012572970008477569,.3744584023952484,.5756075978279114,0,2,9,12,2,3,-1,9,13,2,1,3,-.007424994837492704,.7513859272003174,.4728231132030487,0,2,3,12,1,2,-1,3,13,1,1,2,.0005090812919661403,.540489673614502,.2932321131229401,0,2,9,11,2,3,-1,9,12,2,1,3,-.001280845026485622,.6169779896736145,.4273349046707153,0,2,7,3,3,1,-1,8,3,1,1,3,-.0018348860321566463,.2048496007919312,.5206472277641296,0,2,17,7,3,6,-1,17,9,3,2,3,.0274848695844412,.5252984762191772,.1675522029399872,0,2,7,2,3,2,-1,8,2,1,2,3,.0022372419480234385,.5267782807350159,.2777658104896545,0,2,11,4,5,3,-1,11,5,5,1,3,-.008863529190421104,.69545578956604,.4812048971652985,0,2,4,4,5,3,-1,4,5,5,1,3,.004175397101789713,.4291887879371643,.6349195837974548,0,2,19,3,1,2,-1,19,4,1,1,2,-.0017098189564421773,.2930536866188049,.5361248850822449,0,2,5,5,4,3,-1,5,6,4,1,3,.006532854866236448,.4495325088500977,.7409694194793701,0,2,17,7,3,6,-1,17,9,3,2,3,-.009537290781736374,.3149119913578033,.5416501760482788,0,2,0,7,3,6,-1,0,9,3,2,3,.0253109894692898,.5121892094612122,.1311707943677902,0,2,14,2,6,9,-1,14,5,6,3,3,.0364609695971012,.5175911784172058,.2591339945793152,0,2,0,4,5,6,-1,0,6,5,2,3,.0208543296903372,.5137140154838562,.1582316011190414,0,2,10,5,6,2,-1,12,5,2,2,3,-.0008720774785615504,.5574309825897217,.439897894859314,0,2,4,5,6,2,-1,6,5,2,2,3,-15227000403683633e-21,.5548940896987915,.3708069920539856,0,2,8,1,4,6,-1,8,3,4,2,3,-.0008431650931015611,.3387419879436493,.5554211139678955,0,2,0,2,3,6,-1,0,4,3,2,3,.0036037859972566366,.5358061790466309,.3411171138286591,0,2,6,6,8,3,-1,6,7,8,1,3,-.006805789191275835,.6125202775001526,.4345862865447998,0,2,0,1,5,9,-1,0,4,5,3,3,-.0470216609537601,.2358165979385376,.519373893737793,0,2,16,0,4,15,-1,16,0,2,15,2,-.0369541086256504,.7323111295700073,.4760943949222565,0,2,1,10,3,2,-1,1,11,3,1,2,.0010439479956403375,.5419455170631409,.3411330878734589,0,2,14,4,1,10,-1,14,9,1,5,2,-.00021050689974799752,.2821694016456604,.5554947257041931,0,2,0,1,4,12,-1,2,1,2,12,2,-.0808315873146057,.9129930138587952,.4697434902191162,0,2,11,11,4,2,-1,11,11,2,2,2,-.0003657905908767134,.6022670269012451,.3978292942047119,0,2,5,11,4,2,-1,7,11,2,2,2,-.00012545920617412776,.5613213181495667,.384553998708725,0,2,3,8,15,5,-1,8,8,5,5,3,-.0687864869832993,.2261611968278885,.5300496816635132,0,2,0,0,6,10,-1,3,0,3,10,2,.0124157899990678,.4075691998004913,.5828812122344971,0,2,11,4,3,2,-1,12,4,1,2,3,-.004717481788247824,.2827253937721252,.5267757773399353,0,2,8,12,3,8,-1,8,16,3,4,2,.0381368584930897,.5074741244316101,.1023615971207619,0,2,8,14,5,3,-1,8,15,5,1,3,-.0028168049175292253,.6169006824493408,.4359692931175232,0,2,7,14,4,3,-1,7,15,4,1,3,.008130360394716263,.4524433016777039,.76060950756073,0,2,11,4,3,2,-1,12,4,1,2,3,.006005601957440376,.5240408778190613,.185971200466156,0,3,3,15,14,4,-1,3,15,7,2,2,10,17,7,2,2,.0191393196582794,.5209379196166992,.2332071959972382,0,3,2,2,16,4,-1,10,2,8,2,2,2,4,8,2,2,.0164457596838474,.5450702905654907,.3264234960079193,0,2,0,8,6,12,-1,3,8,3,12,2,-.0373568907380104,.6999046802520752,.4533241987228394,0,2,5,7,10,2,-1,5,7,5,2,2,-.0197279006242752,.2653664946556091,.54128098487854,0,2,9,7,2,5,-1,10,7,1,5,2,.0066972579807043076,.4480566084384918,.7138652205467224,0,3,13,7,6,4,-1,16,7,3,2,2,13,9,3,2,2,.0007445752853527665,.4231350123882294,.5471320152282715,0,2,0,13,8,2,-1,0,14,8,1,2,.0011790640419349074,.5341702103614807,.3130455017089844,0,3,13,7,6,4,-1,16,7,3,2,2,13,9,3,2,2,.0349806100130081,.5118659734725952,.343053013086319,0,3,1,7,6,4,-1,1,7,3,2,2,4,9,3,2,2,.0005685979267582297,.3532187044620514,.5468639731407166,0,2,12,6,1,12,-1,12,12,1,6,2,-.0113406497985125,.2842353880405426,.5348700881004333,0,2,9,5,2,6,-1,10,5,1,6,2,-.00662281084805727,.6883640289306641,.4492664933204651,0,2,14,12,2,3,-1,14,13,2,1,3,-.008016033098101616,.1709893941879273,.5224308967590332,0,2,4,12,2,3,-1,4,13,2,1,3,.0014206819469109178,.5290846228599548,.299338310956955,0,2,8,12,4,3,-1,8,13,4,1,3,-.002780171111226082,.6498854160308838,.4460499882698059,0,3,5,2,2,4,-1,5,2,1,2,2,6,4,1,2,2,-.0014747589593753219,.3260438144207001,.5388113260269165,0,2,5,5,11,3,-1,5,6,11,1,3,-.0238303393125534,.7528941035270691,.4801219999790192,0,2,7,6,4,12,-1,7,12,4,6,2,.00693697901442647,.5335165858268738,.3261427879333496,0,2,12,13,8,5,-1,12,13,4,5,2,.008280625566840172,.458039402961731,.5737829804420471,0,2,7,6,1,12,-1,7,12,1,6,2,-.0104395002126694,.2592320144176483,.5233827829360962,39.1072883605957,80,0,2,1,2,6,3,-1,4,2,3,3,2,.0072006587870419025,.325888603925705,.6849808096885681,0,3,9,5,6,10,-1,12,5,3,5,2,9,10,3,5,2,-.002859358908608556,.5838881134986877,.2537829875946045,0,3,5,5,8,12,-1,5,5,4,6,2,9,11,4,6,2,.0006858052802272141,.5708081722259521,.2812424004077911,0,2,0,7,20,6,-1,0,9,20,2,3,.007958019152283669,.2501051127910614,.5544260740280151,0,2,4,2,2,2,-1,4,3,2,1,2,-.0012124150525778532,.2385368049144745,.5433350205421448,0,2,4,18,12,2,-1,8,18,4,2,3,.00794261321425438,.3955070972442627,.6220757961273193,0,2,7,4,4,16,-1,7,12,4,8,2,.0024630590341985226,.5639708042144775,.2992357909679413,0,2,7,6,7,8,-1,7,10,7,4,2,-.006039659958332777,.218651294708252,.541167676448822,0,2,6,3,3,1,-1,7,3,1,1,3,-.0012988339876756072,.23507060110569,.5364584922790527,0,2,11,15,2,4,-1,11,17,2,2,2,.00022299369447864592,.380411297082901,.572960615158081,0,2,3,5,4,8,-1,3,9,4,4,2,.0014654280385002494,.25101679