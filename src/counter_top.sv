module counter_top (
  input logic clk_i,
  input logic rst_ni,
  input logic start,
  output logic [7:0] pixel_o,
  output logic finished,
  output logic out_valid
);

  logic [9:0] counter;
  logic [9:0] og_to_rot;
  logic [7:0] temp_pix; 

  always_ff @(posedge clk_i or negedge rst_ni) begin 
    if (~rst_ni) begin
	  counter <= '0;
	  pixel_o <= '0; 
	  finished <= 1'b0;
	end else if (counter == 'd784) begin 
	  finished <= 1'b1; 	
	  pixel_o <= '0;
	  counter <= 'd783; 
	end else if (start && counter != 'd784) begin
	  finished <= 1'b0;
	  out_valid <= start;
	  counter <= counter + 1'b1;
	  pixel_o <= temp_pix;
	end 
  end 
  
  rotate_lut_40 r_l (
    .counter(counter),
	.rot_addr(og_to_rot)
  );
   
  mnist_lut m_l (
    .address(og_to_rot),
    .pixel(temp_pix)
  );
endmodule : counter_top