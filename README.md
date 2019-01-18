	_____________________________________________________________________________________________

					Convolutional Robust Principal Component  Analysis   (CORONA)
	_____________________________________________________________________________________________


** Contents

	1. Overview
	2. Requirements
	3. Installation and basic operation
	4. Copyright
	5. Warranty
	6. History
	7. Download
	8. Trademarks

** Publishers
	
	Oren Solomon				orensol@campus.technion.ac.il
	Regev Cohen                 regev.cohen@campus.technion.ac.il
		
	Department of Electrical Engineering
	Technion - Israel Institute of Technology
	Haifa, 32000, Israel

1. Overview

	We present the codes for Convolutional Robust Principal Component  Analysis   (CORONA)
	CORONA is a deep learning based robust PCA network. CORONA's architecture is based on unfolding an iterative algorithm for obtaining a sum of low-rank and sparse matrices representation of the input data. 
    The iterative algorithm is a modified version of th fast iterative soft-shrinkage/thresholding algorithm (FISTA), by Beck and Teboulle, SIIMS, 2009. CORONA is able to achieve exteremly fast convergence compared 
    with FISTA. In the attached codes, CORONA was applied to the separation of contrast signal from hte cluttering tissue in contrast enhanced ultrasound imaging. 	
	
	Descriptions for running the codes are located in the Documentation folder. However, this code comes without any warranty or any additional explanations. Some modifications to file paths' and locations should be made
	prior to running the code. 
	
	This code is for academic purposes only.
	
	If you are using this code, please cite: 
    1.	Solomon, Oren, et al. "Deep Unfolded Robust PCA with Application to Clutter Suppression in Ultrasound", Arxiv.


2. Requirements

	• MATLAB R2016a or newer (previous versions require modifications).
	• Python 3.5.2,  using  the  PyTorch  0.4.1  package.
	• A solid GPU for training the network 
	• At least 8GB RAM; 64GB or more is recommended.


3. Installation and basic operation
	
	To Install:
	-----------
	1. Simply run the codes from the relevant directories. More descriptions are inside the Documentation folder.
	
4. Copyright

    Copyright © 2018 Oren Solomon, Regev Cohen and Yi Zhang, Department of Electrical Engineering, 
	Technion - Israel Institute of Technology, Haifa, 32000, Israel
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
	
	This code is for academic purposes only.
	
5. Warranty

	Any warranty is strictly refused and you cannot anticipate any financial or
	technical support in case of malfunction or damage; see the copyright notice.

	Feedback and comments are welcome. We will try to track reported problems and
	fix bugs.

	Bugs are encouraged to be reported to orensol@campus.technion.ac.il or regev.cohen@campus.technion.ac.il
	
6. History

  • January 18, 2019
	Version 1.0 released under GNU GPL version 3.


7. Download

	The code is available on GitHub: https://github.com/KrakenLeaf/CORONA
	
8. Trademarks

	MATLAB is a registered trademark of The MathWorks. Other product or brand
	names are trademarks or registered trademarks of their respective holders.
	
	All third party software and code packages are disctributed under the GNU license as well. 
	The authors claim no responsibility for this software and code.
