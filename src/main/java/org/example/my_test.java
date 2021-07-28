package djlTest;

import java.io.IOException;
import java.nio.DoubleBuffer;
import java.nio.file.*;
import java.util.*;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.*;


public class my_test {

    public static void main(String[] args) throws IOException, ModelException{
        try(NDManager manager = NDManager.newBaseManager()) {
            //load data
            Path p = Paths.get("build/data/testdata");
            NDList input = manager.load(p);
            List<String> n = new ArrayList<String>();
            n.add("data0");n.add("data1");n.add("data2");n.add("data3");n.add("data4");n.add("data5");
            for(int i =0;i<6;i++){
                input.get(i).setName(n.get(i));
            }
//            NDList newinput = new NDList(input.get(3),input.get(4),input.get(5),input.get(0),input.get(1),input.get(2));
//            input.forEach(nd->System.out.print(nd.getShape()));
//            my_test.predict(nd3);
//            --------------------
            //create data
//            NDManager newmanager = NDManager.newBaseManager();
//            NDArray input1 = newmanager.zeros(new Shape(32,32,745));
//            NDArray input2 = newmanager.ones(new Shape(32,745));
//            NDArray input3 = newmanager.zeros(new Shape(32,24,5));
//            NDArray input4 = newmanager.zeros(new Shape(32,1));
//            NDArray input5 = newmanager.zeros(new Shape(32,1));
//            NDArray input6 = newmanager.zeros(new Shape(32,745,5));
//            NDList input = new NDList(input1,input2,input3,input4,input5,input6);

            //load model
            Path modeDir = Paths.get("build/sy_model/prediction_net");
            Model model = Model.newInstance("DeepAR");
            model.load(modeDir,"prediction_net");

//        Criteria<NDList, NDList> criteria = Criteria.builder()
//                .setTypes(NDList.class, NDList.class) // defines input and output data type
//                .optModelUrls("file:///Users/cuijiahua/code/djl-master/examples/build/sample_model/sample_model") // search models in specified path
////                .optModelPath(Paths.get("/build/sy_model/prediction_net/"))
//                .optEngine("MXNet")
//                .optModelName("sample_model") // specify model file prefix
//                .build();
//
//        ZooModel<NDList, NDList> model = criteria.loadModel();
            Predictor<NDList, NDList> predictor = model.newPredictor(new MyTranslator());
            NDList predictResult = predictor.predict(input);


        } catch (TranslateException e) {
            e.printStackTrace();
        }

    }

//    public static Object predict(NDList input)
//            throws MalformedModelException, ModelNotFoundException, IOException,
//            TranslateException {
//
//        return predictResult;
//
//    }


    public static final class MyTranslator implements Translator<NDList,NDList>{
        @Override
        public NDList processInput(TranslatorContext ctx, NDList input){return input;}
        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list){return list;}
        @Override
        public Batchifier getBatchifier(){return null;}
    }

//        Criteria<NDList,NDList> criteria = Criteria.builder()
//                .setTypes(NDList.class,NDList.class)
////                .optTranslator(translator)
//                .optModelUrls("file:///Users/cuijiahua/code/djl-master/examples/build/sy_model/prediction_net")
//                .optArtifactId("ai.djl.localmodelzoo:prediction_net")
//                .optProgress(new ProgressBar())
//                .build();
//        ZooModel<NDList,NDList> model = ModelZoo.loadModel(criteria);
//        ----------------------


}
