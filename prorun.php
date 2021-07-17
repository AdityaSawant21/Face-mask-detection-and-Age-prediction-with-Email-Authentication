<?php
shell_exec('mkdir folder');
$output= shell_exec('python detect_mask_video.py');
print($output);
?>
