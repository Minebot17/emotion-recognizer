import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class EmotionRecognizer {

    private static final int dataCount = 275;
    private static final float forTrainingPercent = 0.8f;
    private static final int inputElementCount = 256;
    private static final int labelsCount = 4;

    private static Random rnd;
    private static BufferedReader reader;
    private static DataSetIterator iter;
    private static int forTraining;

    public static void main(String[] args) throws IOException, InterruptedException, URISyntaxException {
        rnd = new Random();
        reader = new BufferedReader(new InputStreamReader(System.in));

        System.out.println("What you want to do? (new/train/check/run)");
        String doTypeResponse = reader.readLine();

        forTraining = (int)(dataCount * forTrainingPercent) - 1;
        boolean isCheck = doTypeResponse.equals("check");
        SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, ",");
        featureReader.initialize(new NumberedFileInputSplit("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\%d_input.csv", isCheck ? forTraining : 0, isCheck ? dataCount : forTraining));
        labelReader.initialize(new NumberedFileInputSplit("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\%d_labels.csv", isCheck ? forTraining : 0, isCheck ? dataCount : forTraining));
        iter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, labelsCount, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        switch (doTypeResponse) {
            case "new":
                CreateNewNetwork();
                break;
            case "train":
                TrainExistModel(null);
                break;
            case "check":
                CheckModel();
                break;
            case "run":
                RunPrediction();
                break;
        }
    }

    private static void RunPrediction() throws IOException, InterruptedException, URISyntaxException {
        String fileName = RequestString("Enter input file name without extension: ", "");
        CreatePreudoLabels("./" + fileName + ".csv");
        SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, ",");
        featureReader.initialize(new CollectionInputSplit(new URI[]{ new URI("./" + fileName + ".csv") }));
        labelReader.initialize(new CollectionInputSplit(new URI[]{ new URI("./tmp.csv") }));
        DataSetIterator localIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, labelsCount, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("./model.bin");

        DataSet allData = localIter.next();
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);
        INDArray output = model.output(allData.getFeatures());
        float[] result = new float[labelsCount];
        long maxIndex = output.shape()[2] - 1;
        for (int i = 0; i < labelsCount; i++)
            result[i] = output.getFloat(0, i, (int) 0);

        System.out.println("[ 'счастье', 'возбуждение', 'отвращение', 'сострадание' ]");
        System.out.println(Arrays.toString(result));
    }

    private static void CreatePreudoLabels(String inputPath) throws IOException {
        Scanner sc = new Scanner(Paths.get(inputPath));
        List<String> lines = new ArrayList<>();
        try {
            while (sc.hasNextLine()) {
                sc.next();
                lines.add("0");
            }
        }
        catch (NoSuchElementException ignored){}
        Files.write(Paths.get("./tmp.csv"), lines, StandardCharsets.UTF_8);
    }

    private static int getIndexOfLargest( float[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }

    private static void CheckModel() throws IOException {
        String modelName = RequestString("Enter model name to check: ", "");
        boolean isCheckEpoch = RequestBool("Is check many epochs? (default = false)", false);
        int epochNumber = 1;
        if (isCheckEpoch)
            epochNumber = RequestInt("Enter epoch to check addition: ", 0) + 1;

        float[] accs = new float[epochNumber];
        for (int i = 0; i < epochNumber; i++){
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\models\\" + modelName + (isCheckEpoch && i != 0 ? "_" + (i-1) : "") + ".bin");
            int[] classesCount = new int[labelsCount];
            int[] correctCount = new int[labelsCount];
            int[] notCorrectCount = new int[labelsCount];
            int j = 0;

            for (; j < dataCount - forTraining; j++){
                Scanner sc = new Scanner(Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + (j + forTraining) + "_labels.csv"));
                int currentClass = Integer.parseInt(sc.nextLine());
                classesCount[currentClass]++;

                DataSet allData = iter.next();

                INDArray output = model.output(allData.getFeatures());
                float[] result = new float[labelsCount];
                long maxIndex = output.shape()[2] - 1;
                for (int m = 0; m < labelsCount; m++)
                    result[m] = output.getFloat(0, m, (int) maxIndex);

                int lastElement = getIndexOfLargest(result);
                System.out.println(currentClass);
                System.out.println(Arrays.toString(result));
                if (lastElement == currentClass)
                    correctCount[lastElement]++;
                else
                    notCorrectCount[lastElement]++;
            }

            float[] percent = new float[labelsCount];
            for (int n = 0; n < labelsCount; n++){
                percent[n] = correctCount[n] / (float) classesCount[n];
            }

            System.out.println("Result for epoch " + (i - 1) + ":");
            System.out.println("Not correct: " + Arrays.toString(notCorrectCount));
            System.out.println("Correct: " + Arrays.toString(correctCount));
            System.out.println("Count: " + Arrays.toString(classesCount));
            System.out.println("Percent: " + Arrays.toString(percent));

            float sum = 0;
            for (int n = 0; n < labelsCount; n++)
                sum += percent[n];

            System.out.println("Acc: " + (sum/(float)labelsCount));
            accs[i] = (sum/(float)labelsCount);
            iter.reset();
        }

        System.out.println("Index of max acc: " + (getIndexOfLargest(accs) - 1));
    }

    private static void TrainExistModel(String existModel) throws IOException {
        if (existModel == null){
            System.out.println("Enter model name to train: ");
            existModel = reader.readLine();
        }

        int trainEpoch = RequestInt("Enter epoch to train (default = 1): ", 1);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\models\\" + existModel + ".bin");
        for (int i = 0; i < trainEpoch; i++){
            System.out.println("Start epoch number " + i);
            ShuffleTrainData();

            for (int j = 0; j < forTraining; j++){
                try {
                    DataSet allData = iter.next();
                    System.out.println("Training " + (int)(j/(float)forTraining * 100f) + "%");
                    model.fit(allData);
                }
                catch (IllegalStateException ignored){
                    System.out.println("Error at i dataset");
                }
            }

            System.out.println("Epoch " + i + " is complete. Save model as " + existModel + "_" + i);
            ModelSerializer.writeModel(model, "C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\models\\" + (existModel + "_" + i) + ".bin", true);
            iter.reset();
        }

        System.out.println("Training all epochs is complete");
    }

    private static void CreateNewNetwork() throws IOException {
        double learningRate = RequestDouble("Learning rate (default = 0.0001): ", 0.0001d);
        int neuronsCount = RequestInt("Neurons in hidden layer (default = 2048): ", 2048);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .seed(System.currentTimeMillis())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(inputElementCount)
                        .nOut(neuronsCount)
                        .forgetGateBiasInit(1)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DropoutLayer(0.2f))
                .layer(2, new LSTM.Builder()
                        .nIn(neuronsCount)
                        .nOut(neuronsCount)
                        .forgetGateBiasInit(1)
                        .activation(Activation.TANH)
                        .build())
                .layer(3, new DropoutLayer(0.2f))
                .layer(4, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(neuronsCount)
                        .nOut(labelsCount)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        String modelName = RequestString("Enter model name (default model_<randomNumber>): ", "model_" + rnd.nextInt());
        String savePath = "C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\models\\" + modelName + ".bin";
        ModelSerializer.writeModel(model, savePath, true);
        TrainExistModel(modelName);
    }

    private static void ShuffleTrainData() throws IOException {
        for (int i = 0; i < 1000; i++)
        {
            int i1 = rnd.nextInt(forTraining);
            int i2 = rnd.nextInt(forTraining);

            if (i1 == i2)
                continue;

            Files.move(Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "_input.csv"), Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "[_input.csv"));
            Files.move(Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "_labels.csv"), Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "[_labels.csv"));
            Files.move(Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i2 + "_input.csv"), Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "_input.csv"));
            Files.move(Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i2 + "_labels.csv"), Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "_labels.csv"));
            Files.move(Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "[_input.csv"), Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i2 + "_input.csv"));
            Files.move(Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i1 + "[_labels.csv"), Paths.get("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\" + i2 + "_labels.csv"));
        }
    }

    private static String RequestString(String message, String defaultValue) throws IOException {
        System.out.println(message);
        String result = defaultValue;
        String response = reader.readLine();
        if (response != null && !response.equals(""))
            result = response;

        System.out.println();
        return result;
    }

    private static double RequestDouble(String message, double defaultValue) throws IOException {
        System.out.println(message);
        double result = defaultValue;
        String response = reader.readLine();
        if (response != null && !response.equals(""))
            result = Double.parseDouble(response);

        System.out.println();
        return result;
    }

    private static int RequestInt(String message, int defaultValue) throws IOException {
        System.out.println(message);
        int result = defaultValue;
        String response = reader.readLine();
        if (response != null && !response.equals(""))
            result = Integer.parseInt(response);

        System.out.println();
        return result;
    }

    private static boolean RequestBool(String message, boolean defaultValue) throws IOException {
        System.out.println(message);
        boolean result = defaultValue;
        String response = reader.readLine();
        if (response != null && !response.equals(""))
            result = Boolean.parseBoolean(response);

        System.out.println();
        return result;
    }
}
