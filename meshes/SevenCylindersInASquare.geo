SetFactory("OpenCASCADE");

L = 0.1;
H = 0.1;
r = 0.11*H;

lc = 0.01*H;

//Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
//Point(3) = {L, H, 0};
Point(4) = {0, H, 0};

Circle(17) = {0,0,0,0.75*r,0,0.5*Pi};
Circle(18) = {L,H,0,0.75*r,Pi,1.5*Pi};


Line(1) = {6, 2};
Line(2) = {2, 7};
Line(3) = {8, 4};
Line(4) = {4, 5};



Circle(10) = {0.15*L, 0.40*H, 0, 0.9*r, 0, 2*Pi};
Circle(12) = {0.32*L, 0.18*H, 0, 1.2*r, 0, 2*Pi};
Circle(13) = {0.75*L, 0.25*H, 0, 2.0*r, 0, 2*Pi};
Circle(11) = {0.42*L, 0.58*H, 0, 1.6*r, 0, 2*Pi};
Circle(14) = {0.17*L, 0.83*H, 0, 1.2*r, 0, 2*Pi};
Circle(15) = {0.82*L, 0.70*H, 0, 1.2*r, 0, 2*Pi};
Circle(16) = {0.62*L, 0.85*H, 0, 0.9*r, 0, 2*Pi};

Curve Loop(21) = {1, 2, 18, 3, 4, 17};
Curve Loop(22) = {13};
Curve Loop(23) = {11};
Curve Loop(24) = {15};
Curve Loop(25) = {14};
Curve Loop(26) = {12};
Curve Loop(27) = {16};
Curve Loop(20) = {10};

Plane Surface(31) = {21, 22, 23, 24, 25, 26, 27, 20};

// No need for the physical points since we are using RT
//Inlet
//Physical Point("inlet", 41) = {1,4};
Physical Line("inlet", 41) = {17};

//Outlet
//Physical Point("outlet", 42) = {2,3};
Physical Line("outlet", 42) = {18};

//Walls
Physical Line("walls", 43) = {1,2,3,4};
//Physical Line("walls", 43) = {1,5,3,6};

//Cylinders
Physical Line("cylinders",44) = {10,11,12,13,14,15,16};

// Domain
Physical Surface("fluid",51) = {31};

Mesh.Algorithm = 2;
Mesh.ScalingFactor = 1.0;
//Mesh.MinimumCirclePoints = 10;
Mesh.Smoothing = 2;

