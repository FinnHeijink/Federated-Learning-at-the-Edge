module conv_tb;
  logic clk_i, rst_ni, pix_data_valid, conv_finished;
  logic [7:0] pixel_i, pixel_o;
  logic [143:0] k_val;
    
  top_conv CNN (
    .k_val(k_val),
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .pixel_i(pixel_i),
    .pix_data_valid(pix_data_valid),
    .pixel_o(pixel_o),
    .conv_finished(conv_finished)
  );

  always #50 clk_i = ~clk_i;   

  initial begin
    rst_ni = 0; 
	clk_i = 0;
	pix_data_valid = 0;
	pixel_i = '0; 
    k_val = 144'b000000000000000100000000000000100000000000000001000000000000001000000000000001000000000000000010000000000000000100000000000000100000000000000001;
    #20 rst_ni = 1; 
  end

  integer data_file;
  integer scan_file; 
  logic [7:0] captured_data;
  integer out_file;
  
  initial begin
      data_file = $fopen("output.txt", "r");
      out_file = $fopen("pic_out.txt", "w");
      if (data_file == 0) begin
        $display("data_file handle was NULL");
        $finish;
      end
  end

    always @(posedge clk_i) begin
      if (rst_ni) begin
	      pix_data_valid <= 1'b1;
          scan_file = $fscanf(data_file, "%d\n", captured_data); 
          if (!$feof(data_file)) begin
            //use captured_data as you would any other wire or reg value;
            pixel_i <= captured_data; 
          end else begin
		    pix_data_valid <= 1'b0;
		  end
      end
    end
  
  always @(posedge clk_i) begin
     if (conv_finished == 1) begin
        $fwrite(out_file, "%d\n", pixel_o);         
     end
  end
  
endmodule : conv_tb