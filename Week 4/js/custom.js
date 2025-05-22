const PATH_TO_SET = "..";
(function() { 
    const isQuizBuilderPage = !!document.getElementById("loadQuizBuilder");
    if(!isQuizBuilderPage){
        // run conditional load to see if page has advanced interactives
        /**** Needs to be an absolute path when ready to publish .. **/
        const src0 =PATH_TO_SET + "/assets/advInteractives_pkg/js/init.js";
        document.write('<script src="' + src0 + '"><\/script>');

        // console.log("Trying to load condLoad js file");
        const src1 =PATH_TO_SET + "/assets/advInteractives_pkg/js/condLoad.js";
        document.write('<script src="' + src1 + '"><\/script>');
    }
})();