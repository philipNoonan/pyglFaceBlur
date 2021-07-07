#version 150

    in vec2 outTexCoords;
    out vec4 outColor;
	
    uniform sampler2D samplerTex;


    void main()
    {

        vec4 col = textureLod(samplerTex, outTexCoords, 0);


		outColor = col;
    }