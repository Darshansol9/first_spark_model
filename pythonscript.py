from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("Titanic Data").getOrCreate()

df = spark.read.format("csv").option("header", "true").load("train.csv")

df.show(5)
from pyspark.sql.functions import col
import pyspark.sql.functions as F

dataset = df.select(
    col("Survived").cast("float"),
    col("Pclass").cast("float"),
    col("Sex"),
    col("Age").cast("float"),
    col("Fare").cast("float"),
    col("Embarked"),
)

dataset = dataset.replace("?", None).dropna(how="any")

gender_model = StringIndexer(inputCol="Sex", outputCol="Gender", handleInvalid="keep")
embark_model = StringIndexer(
    inputCol="Embarked", outputCol="Boarded", handleInvalid="keep"
)
required_features = ["Pclass", "Age", "Fare", "Gender", "Boarded"]
assembler = VectorAssembler(inputCols=required_features, outputCol="features")
rf = RandomForestClassifier(labelCol="Survived", featuresCol="features", maxDepth=5)


evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", predictionCol="prediction", metricName="accuracy"
)
training_data, test_data = dataset.randomSplit([0.8, 0.2])


pipeline = Pipeline(stages=[gender_model, embark_model, assembler, rf])

model = pipeline.fit(training_data)

predictions = model.transform(test_data)

accuracy = evaluator.evaluate(predictions)

print(accuracy)

accuracy_in_sample = evaluator.evaluate(model.transform(training_data))

print(accuracy_in_sample)

training_data.select(F.mean(col("Survived"))).show()
