#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Text:: *)
(*Generate shape*)


(* ::Input::Initialization:: *)
datasetsDir = "datasets";
stepSize = 64;
kernels = 4;
samplesNum = 1000;
outdir = StringForm["/home/ivan/python/deepcfd/``/dataset_rndshap_Randombc_move_body_step_1to``", datasetsDir, stepSize]

With[{dirname = FileNameJoin[{ToString[StringForm["``", outdir]]}]}, 
    Switch[FileType[dirname], None, CreateDirectory[dirname], Directory, 
    Null, File, Print["File with same name already exists!!"]]];


(* ::Input::Initialization:: *)
tablePrec = 5;
(*maxCellMeasure = 0.02;*)
maxCellMeasure={"Length"->0.098,"Area"->0.039}; 
(*"MaxBoundaryCellMeasure" \[Rule] 1*)
meshQualityGoal = "Maximal";


(* ::Text:: *)
(*Polygon mesh*)


(* ::Input::Initialization:: *)
L=10;Lc=2;Lp=L-Lc;H=2;n=8;
minR=0.25;

amax=1;
bmax= 0.5;
Shift\[Phi]Range = {0.55, 1.45};
\[Phi]arr=Table[(i-1)/n 2 \[Pi],{i,n+1}];

ShiftXRange = {-1.0, 4.0};
ShiftYRange = {-0.5, 0.5};

step = 1/stepSize;
prec = 6;
acc = 6;


(* ::Text:: *)
(*Equations for u, v, p, T*)


(* ::Input::Initialization:: *)
Ro = 1; \[Theta] = 0.5; nu = 1/100;
navierstokes = { 
Inactive[Div][(-nu Inactive[Grad][u[x, y], {x, y}]), {x, y}] + {u[x, y], v[x, y]} . Inactive[Grad][u[x, y], {x, y}] + 1/Ro \!\(\*SuperscriptBox[\(p\), 
TagBox[
RowBox[{"(", 
RowBox[{"1", ",", "0"}], ")"}],
Derivative],
MultilineFunction->None]\)[x, y],
Inactive[Div][(-nu Inactive[Grad][v[x, y], {x, y}]), {x, y}] + {u[x, y], v[x, y]} . Inactive[Grad][v[x, y], {x, y}] + 1/Ro \!\(\*SuperscriptBox[\(p\), 
TagBox[
RowBox[{"(", 
RowBox[{"0", ",", "1"}], ")"}],
Derivative],
MultilineFunction->None]\)[x, y],
\!\(\*SuperscriptBox[\(u\), 
TagBox[
RowBox[{"(", 
RowBox[{"1", ",", "0"}], ")"}],
Derivative],
MultilineFunction->None]\)[x, y] + \!\(\*SuperscriptBox[\(v\), 
TagBox[
RowBox[{"(", 
RowBox[{"0", ",", "1"}], ")"}],
Derivative],
MultilineFunction->None]\)[x, y],
Inactive[Div][(-\[Theta] Inactive[Grad][T[x, y], {x, y}]), {x, y}] +u[x,y]* \!\(\*SuperscriptBox[\(T\), 
TagBox[
RowBox[{"(", 
RowBox[{"1", ",", "0"}], ")"}],
Derivative],
MultilineFunction->None]\)[x, y] + v[x,y]*\!\(\*SuperscriptBox[\(T\), 
TagBox[
RowBox[{"(", 
RowBox[{"0", ",", "1"}], ")"}],
Derivative],
MultilineFunction->None]\)[x, y]
};


(* ::Text:: *)
(*Border conditions*)


(* ::Input::Initialization:: *)
Urange={0.5, 1};
Trange = {0, 1};
crange = {0,0.5};
krange1 = {0, 3};
krange2 = {3, 6};
T0 = 0;
T1 = 1;
NanVal = -50;


(* ::Input::Initialization:: *)
Needs["NDSolve`FEM`"]


(* ::Input::Initialization:: *)
SetSystemOptions["CacheOptions"->{"Symbolic"->{"Cache"->False}}];


calcsPerKernel = 2;
start = 0;
stop = samplesNum;

LaunchKernels[kernels];
Print[MemoryInUse[]];
totalIters = Round[(stop - start) / kernels]
Do[
    With[
        {}
        ,
        ParallelEvaluate[
            kernelId = Mod[$KernelID - 1, kernels];
            idx = start + kernels * (i - 1) + kernelId;
            numStr = NumberForm[idx, 5, NumberPadding -> {"0", ""}];
            Print["KernelID #", $KernelID, ": idx = ", numStr, ", CALC EQ"];
            
			ShiftX = RandomReal[ShiftXRange];
			ShiftY = RandomReal[ShiftYRange];

			a=RandomReal[{0.4,amax}];
			b=RandomReal[{0.4,bmax}];
			
			PointsAR=Table[{RandomReal[{minR,1}],\[Phi]arr[[i]]+(2 \[Pi])/n*RandomReal[Shift\[Phi]Range]},{i,n}];
			PointsXY=Table[{a*PointsAR[[i]][[1]]*Cos[PointsAR[[i]][[2]]]+ShiftX,b*PointsAR[[i]][[1]]*Sin[PointsAR[[i]][[2]]]+ShiftY},{i,n}];
			
            pol = BSplineCurve[PointsXY, SplineClosed -> True];
            g = FilledCurve @ pol;
            objreg = g;
            pol = DiscretizeGraphics[Graphics[g]];
            pol\[CapitalOmega] = RegionDifference[Rectangle[{-Lc, -(H / 2)}, {Lp, H / 2}], pol];
            pol\[CapitalOmega] = BoundaryDiscretizeRegion[pol\[CapitalOmega], AccuracyGoal -> 6];
            polMesh = ToElementMesh[pol\[CapitalOmega], MaxCellMeasure -> maxCellMeasure, MeshQualityGoal -> meshQualityGoal, "MeshOrder" -> 1];
           
            U1 = 3/2 * RandomReal[Urange];
            T2 = RandomReal[Trange];
            c1 = RandomReal[crange];
            c2 = RandomReal[crange / 2];
            k1 = RandomReal[krange1] * \[Pi];
            k2 = RandomReal[krange2] * \[Pi];
            
            inflowBC = DirichletCondition[{u[x, y] == U1 * (1 - (y / (H / 2)) ^ 2) * ((1 + c1 * Sin[y * k1]) + (1 + c2 * Sin[y * k2])), v[x, y] == 0}, x == -Lc];
            outflowBC = DirichletCondition[p[x, y] == 0, x == Lp];
            wallflowBC = DirichletCondition[{u[x, y] == 0, v[x, y] == 0}, -Lc < x < Lp];
            objflowBC = DirichletCondition[{u[x, y] == 0, v[x, y] == 0}, {x, y} \[Element] objreg];
            intempBC = DirichletCondition[T[x, y] == T0 + (T1 - T0) / H * (y + H / 2), x == -Lc || y == -H / 2 || y == H / 2 || x == Lp];
            objtempBC = DirichletCondition[T[x, y] == T2, {x, y} \[Element] objreg];
            bcs = {inflowBC, outflowBC, wallflowBC, objflowBC, intempBC, objtempBC};
            
	        sol = Check[
	            {polUVel, polVVel, polPres, polTemp} = NDSolveValue[
		            {navierstokes == {0, 0, 0, 0}, bcs},
		            {u, v, p, T},
		            {x, y} \[Element] pol\[CapitalOmega],
		            Method -> {"FiniteElement", "InterpolationOrder" -> {u -> 2, v -> 2, p -> 1, T -> 1}, "MeshOptions" -> {"MaxCellMeasure" -> maxCellMeasure}}
		            ],
	            $Failed,
	            {NDSolveValue::fempsf}
	            ];
			
			If[
				sol === $Failed,
				Print["WARNING: SKIP ", idx, "; ERROR in NDSolveValue"];
				Continue[]
			];
            (*BCRegionUV = RegionUnion[Line[{{L-Lc, -H/2},{-Lc, -H/2},{-Lc, H/2},{L-Lc, H/2}}],pol];
            BCRegionP = Line[{{L-Lc, -H/2},{ L-Lc, H/2}}];
            BCRegionT = RegionUnion[Line[{{-Lc, -H/2},{-Lc, H/2}}],pol];
            BCRegionTExtended = RegionUnion[Line[{{L-Lc, -H/2},{-Lc, -H/2},{-Lc, H/2},{L-Lc, H/2}}],pol];*)
                
            Print["KernelID #", $KernelID, ": idx = ", numStr, ", SAVE NODES"];
            polUVeltab = Table[SetAccuracy[polUVel[x, y], acc], {y, - (H / 2) + step, (H / 2), step}, {x, -Lc + step, Lp, step}];
            polVVeltab = Table[SetAccuracy[polVVel[x, y], acc], {y, - (H / 2) + step, (H / 2), step}, {x, -Lc + step, Lp, step}];
            polPrestab = Table[SetAccuracy[polPres[x, y], acc], {y, - (H / 2) + step, (H / 2), step}, {x, -Lc + step, Lp, step}];
            polTemptab = Table[SetAccuracy[polTemp[x, y], acc], {y, - (H / 2) + step, (H / 2), step}, {x, -Lc + step, Lp, step}];
            
            (*polUVeltab=Table[SetPrecision[polUVel[x,y], prec],{y,-( H/2)+step,(H/2),step}, {x,-Lc+step,Lp,step}];*)
            (*polUVeltab=Table[SetPrecision[polUVel[x,y], prec],{y,-( H/2),(H/2),step}, {x,-Lc,Lp,step}];*)
            
            Export[ToString[StringForm["``/``_pol_UVel.csv", outdir, numStr]], polUVeltab, "CSV"];
            Export[ToString[StringForm["``/``_pol_VVel.csv", outdir, numStr]], polVVeltab, "CSV"];
            Export[ToString[StringForm["``/``_pol_Pres.csv", outdir, numStr]], polPrestab, "CSV"];
            Export[ToString[StringForm["``/``_pol_Temp.csv", outdir, numStr]], polTemptab, "CSV"];
            
            Print["KernelID #", $KernelID, ": idx = ", numStr, ", SAVE BOUNDARY CONDITIONS AND LABELS" ];
            polLabeltab = Table[Boole[RegionMember[pol, {x, y}]], {y, -(H / 2) + step, (H / 2), step}, {x, -Lc + step, Lp, step}];
            BCs = SetPrecision[{U1, T2, c1, c2, k1, k2}, prec];
            
            Export[ToString[StringForm["``/``_pol_Label.csv", outdir, numStr]], polLabeltab, "CSV"];
            Export[ToString[StringForm["``/``_pol_BCs.csv", outdir, numStr ]], BCs, "CSV"];
            
            ClearAll[polUVel, polVVel, polPres, polTemp, polUVeltab, polVVeltab, polPrestab, polTemptab, Labeltab, BCs];
            ClearSystemCache[];
            (*Print["MemoryInUse : "];
            Print[MemoryInUse []];*)
        ]
    ];
    If[Mod[i, calcsPerKernel] == 0,
	    CloseKernels[];
	    ClearSystemCache[];
	    LaunchKernels[kernels];
    ];
    ,
    {i, totalIters}
];

CloseKernels[];
ClearSystemCache[];
Print[MemoryInUse[]];





