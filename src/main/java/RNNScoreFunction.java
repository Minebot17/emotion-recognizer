import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Arrays;

public class RNNScoreFunction extends EvaluationScoreFunction {

    private final int labels = 4;

    public RNNScoreFunction(Evaluation.Metric metric){
        super(metric);
    }

    public double score(MultiLayerNetwork net, DataSetIterator iterator) {
        int iterations = 0;
        float result = 0;
        if (iterator.hasNext())
        {
            DataSet data = iterator.next();
            INDArray output = net.output(data.getFeatures());
            float sum = 0;
            long maxIndex = output.shape()[2] - 1;
            for (int i = 0; i < labels; i++)
                sum += output.getFloat(0, i, (int) maxIndex);

            sum /= (float) labels;
            result += sum;
            iterations++;
        }

        result /= (float) iterations;
        return result;
    }
}
