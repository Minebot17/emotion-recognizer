import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Properties;

public class EEGDataSource implements DataSource {

    private SequenceRecordReaderDataSetIterator trainIter;
    private SequenceRecordReaderDataSetIterator testIter;

    public EEGDataSource() throws IOException, InterruptedException {
        {
            SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
            SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, ",");
            featureReader.initialize(new NumberedFileInputSplit("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\%d_input.csv", 0, 206));
            labelReader.initialize(new NumberedFileInputSplit("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\%d_labels.csv", 0, 206));
            trainIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        }
        {
            SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
            SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, ",");
            featureReader.initialize(new NumberedFileInputSplit("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\%d_input.csv", 206, 275));
            labelReader.initialize(new NumberedFileInputSplit("C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\out\\%d_labels.csv", 206, 275));
            testIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        }
    }

    @Override
    public void configure(Properties properties) {

    }

    @Override
    public Object trainData() {
        return trainIter;
    }

    @Override
    public Object testData() {
        return testIter;
    }

    @Override
    public Class<?> getDataType() {
        return SequenceRecordReaderDataSetIterator.class;
    }
}
