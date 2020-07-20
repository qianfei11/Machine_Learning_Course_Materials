TP = 0;
FN = 0;
FP = 0;
TN = 0
for i = 1 : length(yPred)
    if yTraining(i) == 1
        if yPred(i) == 1 
            TP = TP + 1;
        else
            FN = FN + 1;
        end;
    else
        if yPred(i) == 1 
            FP = FP + 1;
        else 
            TN = TN + 1;
        end
    end
end
