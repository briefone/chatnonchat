<?php
// ini_set('display_errors', 1);
// ini_set('display_startup_errors', 1);
// error_reporting(E_ALL);

header('Content-Type: application/javascript');

try {
    // Connect to the SQLite database
    $db = new PDO('sqlite:chatnonchat.db');
    $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // Query to return the id and the whole JSON blob
    $sql = "SELECT id, data FROM json_data ORDER BY id DESC LIMIT 100";
    $result = $db->query($sql);

    $data = [];
    foreach ($result as $row) {
      $row_data = json_decode($row['data'], true);
      $row_data['id'] = $row['id'];
      $data[] = $row_data;
    }

    // Encode the data as JSON
    $json_data = json_encode([
      'status' => 'success',
      'data' => $data
  ]);

    // Return the JSON data within a function call
    echo "loaditems($json_data);";

} catch (PDOException $e) {
    // Return the error message as JSON within a function call
    $error_data = json_encode([
        'status' => 'error',
        'message' => $e->getMessage()
    ]);
    echo "loaditems($error_data);";
}



// Delete JSON data from the database 

// try {
//     $db = new PDO('sqlite:chatnonchat.db');
//     $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

//     $stmt = $db->prepare("DELETE FROM json_data WHERE id = :id");
//     $stmt->bindParam(':id', $id);

//     $id = 1;
//     $stmt->execute();

//     echo "JSON data deleted successfully!";
// } catch (PDOException $e) {
//     echo "Failed to delete JSON data: " . $e->getMessage();
// }





 ?>
