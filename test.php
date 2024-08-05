<?php

ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

$response = [
  'class' => 'sadasd',
  'confidence' => 'sadasd',
  'probabilities' => 'sadasd',
  'raw_scores' => 'sadasd',
  'file_path' => 'sadasd'
];

try {
  $db = new PDO('sqlite:chatnonchat.db');
  $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

  $stmt = $db->prepare("INSERT INTO json_data (data) VALUES (:data)");
  $stmt->bindParam(':data', $jsonData);

  $jsonData = json_encode($response);
  $stmt->execute();

  echo "JSON data inserted successfully!";
} catch (PDOException $e) {
  echo "Failed to insert JSON data: " . $e->getMessage();
}

echo json_encode($response);



?>
