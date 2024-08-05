<?php

// ini_set('display_errors', 1);
// ini_set('display_startup_errors', 1);
// error_reporting(E_ALL);

header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header("Access-Control-Allow-Headers: X-Requested-With");

header('Content-Type: application/json');

$maxFileSize = 55 * 1024 * 1024; // 55MB
$base_dir = '/var/www/html/work/chatnonchat';
$upload_dir = './uploads/';
$tmp_dir = './tmp/';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_FILES['file'])) {
        handleFileUpload();
    } else {
        echo json_encode(['error' => 'Invalid request. No file data received.']);
    }
} else {
    echo json_encode(['error' => 'Invalid request method.']);
}

function parseModelName($model) {
    // Remove the prefix 'catnotchat_'
    $model = preg_replace('/^catnotcat_/', '', $model);
    
    // Remove the suffix '_round_X.pth'
    $model = preg_replace('/_round_\d+\.pth$/', '', $model);
    
    return $model;
}

function handleFileUpload() {
    global $maxFileSize, $upload_dir, $tmp_dir, $base_dir;

    if ($_FILES['file']['size'] > $maxFileSize) {
        echo json_encode(['error' => 'File size exceeds the maximum limit of 55MB.', 'code' => 99]);
        exit;
    }

    if (isset($_POST['model'])) {
        $model = basename($_POST['model']);
    } else {
        $model = 'catnotcat_resnet18_SGD_StepLR_round_2.pth';
    }

    // Create temporary directory if it doesn't exist
    if (!is_dir($tmp_dir)) {
        if (!mkdir($tmp_dir, 0775, true)) {
            echo json_encode(['error' => 'Failed to create temporary directory.']);
            exit;
        }
    }

    // Save the file to the temporary directory
    $tmp_file_path = $tmp_dir . basename($_FILES['file']['tmp_name']);
    if (!move_uploaded_file($_FILES['file']['tmp_name'], $tmp_file_path)) {
        echo json_encode(['error' => 'Failed to move file to temporary directory.']);
        exit;
    }

    // Get the file extension from the temporary file
    $file_info = new finfo(FILEINFO_MIME_TYPE);
    $mime_type = $file_info->file($tmp_file_path);
    $file_extension = getExtensionFromMimeType($mime_type);
    if (!$file_extension) {
        echo json_encode(['error' => 'Unsupported file type.']);
        exit;
    }

    // Create a new file name with prefix, model name, and timestamp
    $timestamp = time();
    $file_name_with_extension = 'chatnotchat_' . parseModelName($model) . '_' . $timestamp . '.' . $file_extension;
    $file_path = $upload_dir . $file_name_with_extension;

    // Move the file from temporary directory to the final upload directory
    if (!rename($tmp_file_path, $file_path)) {
        echo json_encode(['error' => 'Failed to move file to upload directory.']);
        exit;
    }

    processImage($file_path, $file_name_with_extension, $model);
}

function getExtensionFromMimeType($mime_type) {
    $mime_map = [
        'image/jpeg' => 'jpg',
        'image/png' => 'png',
        'image/gif' => 'gif',
        'image/webp' => 'webp',
        'image/bmp' => 'bmp',
        'image/heic' => 'heic',
        'image/heif' => 'heif',
        // Add more MIME types and their corresponding extensions as needed
    ];

    return $mime_map[$mime_type] ?? null;
}

function processImage($file_path, $file_name_with_extension, $model) {
    global $base_dir, $upload_dir;

    $model_path = $base_dir . '/' . $model;
    $script_path = $base_dir . '/catnotcat.py';
    $venv_path = $base_dir . '/venv';  // Path to your virtual environment

    // Escape shell command and execute it
    $command = escapeshellcmd("$venv_path/bin/python3 $script_path --image '$file_path' --model $model_path");
    $output = shell_exec($command);

    // Check for errors in the output
    if (json_decode($output) === null && json_last_error() !== JSON_ERROR_NONE) {
        $response = ['error' => 'An error occurred while processing the image.', 'details' => $output];
        echo json_encode($response);
    } else {
        // Parse the output from the Python script
        $data = json_decode($output, true);

        if (isset($data['class'])) {
            $classification = $data['class'];
            $destination_dir = $upload_dir . ($classification === 'Cat' ? 'cat/' : 'notcat/');

            // Create the destination directory if it doesn't exist
            if (!is_dir($destination_dir)) {
                if (!mkdir($destination_dir, 0775, true)) {
                    echo json_encode(['error' => 'Failed to create classification directory.']);
                    exit;
                }
            }

            // Move the file to the appropriate directory
            $new_file_path = $destination_dir . $file_name_with_extension;
            if (rename($file_path, $new_file_path)) {
                $response = [
                    'class' => $data['class'],
                    'confidence' => $data['confidence'],
                    'probabilities' => $data['probabilities'],
                    'raw_scores' => $data['raw_scores'],
                    'file_path' => $new_file_path,
                    'model' => $model,
                ];

                try {
                    $db = new PDO('sqlite:chatnonchat.db');
                    $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
                
                    $stmt = $db->prepare("INSERT INTO json_data (data) VALUES (:data)");
                    $stmt->bindParam(':data', $jsonData);
                
                    $jsonData = json_encode($response);
                    $stmt->execute();
                
                    // echo "JSON data inserted successfully!";
                } catch (PDOException $e) {
                    // echo "Failed to insert JSON data: " . $e->getMessage();
                }

                echo json_encode($response);
            } else {
                echo json_encode(['error' => 'Failed to move file to classification directory.']);
            }
        } else {
            echo json_encode(['error' => 'Classification result missing from Python script output.', 'details' => $output]);
        }
    }
}

?>
